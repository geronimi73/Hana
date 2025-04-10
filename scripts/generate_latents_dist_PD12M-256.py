import torch.distributed as dist, torch, random, time, os, json, pdb, asyncio

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from diffusers import AutoencoderDC
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import make_grid, PIL_to_latent, latent_to_PIL, pil_add_text
from utils_preprocess import resizeToClosestMultOf, pad, acquire_lock, release_lock, url_to_image, download_all_images

def process_dcae_batch(batch, dcae):
    labels = batch["labels"]
    images = batch["images"]
    image_ids = batch["image_ids"]
    # bfloat16 latents 
    latents = PIL_to_latent(images, dcae).to(torch.bfloat16).cpu()   
    
    return [
        dict(
        	image_id=image_ids[i], 
        	label=labels[i], 
        	latent=latents[None,i],
        	latent_shape=latents[i].shape
        )
        for i in range(len(labels))
    ]

def process(rank, is_master, world_size):
	test_run = False
	dtype = torch.bfloat16
	device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
	source_dataset = "PD12M"
	hf_dataset = "g-ronimo/PD12M-256px_dc-ae-f32c32-sana-1.0"
	resizeTo = 256
	download_batch_size = 32  
	dcae_batch_size = 32  # don't set this too high to keep the GPU busy
	dcae_f = 32  
	upload_every = 100_000 if not test_run else 200
	splits=["train", "test"]
	col_imgid="id"
	col_url="url"
	col_label="caption"
	zerobatch_saved = False # ignore this

	print("Loading models","rank", rank)
	model = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
	dcae = AutoencoderDC.from_pretrained(model, subfolder="vae", torch_dtype=dtype).to(device)

	print(f"Loading dataset {source_dataset}","rank", rank)
	ds = load_from_disk(source_dataset)
	# Spawning/PD12M was first split into test/train and saved to disk like this:
	# ds = load_dataset("Spawning/PD12M", cache_dir="~/ssd-2TB/hf_cache")
	# ds = ds["train"].train_test_split(shuffle=True, seed=42, test_size=50_000)
	# ds.save_to_disk("PD12M")

	for split in splits:
		dataset_list = []   # list of dicts per AR bucket, each containing list of {label=.., latent=..}
		parts_uploaded = 0   
		dcae_batches = {}   # buffer, collect samples and batch process when full
		samples_uploaded = 0
		samples_failed = 0

		if test_run:
			ds[split] = ds[split].select(range(1_000))

		# Poor man's distributed sampler
		indices = list(range(len(ds[split])))  # all ranks
		indices = indices[rank : len(indices) : world_size] # subsample for current rank

		# DataLoader for prefetching samples - will download and resize in advance
		def collate_fn(indices):
			imgids = [ds[split][idx][col_imgid] for idx in indices]
			labels = [ds[split][idx][col_label].strip() for idx in indices]
			urls = [ds[split][idx][col_url] for idx in indices]
			images = asyncio.run(download_all_images(urls))
			# resize images
			images = [
				# careful, img might be none if download failed
				resizeToClosestMultOf(img, resizeTo=resizeTo, denominator=dcae_f, maxDim=2*resizeTo) if img else None
				for img in images
			]

			return indices, imgids, images, labels
		dl = DataLoader(indices, collate_fn=collate_fn,
			batch_size=download_batch_size,
			prefetch_factor=10, 
			num_workers=5,
		)

		pbar = tqdm(total=len(indices), desc=f"Processing split {split}, rank {rank}")

		# Download one batch, batch = (indices_batch, imgids, images, labels)
		for batch in dl:
			# Loop through items in batch one by one
			for idx, imgid, img, label in zip(*batch):
				# failed download
				if img is None: continue
				
				# Put image into queue for DC-AE encoding to latent
				ar_bucket = img.size # legacy caused naming, not really an aspect ratio bucket
				if not ar_bucket in dcae_batches: 
					dcae_batches[ar_bucket]={"labels": [], "images": [], "image_ids": []}
				dcae_batches[ar_bucket]["labels"].append(label)
				dcae_batches[ar_bucket]["images"].append(img)
				dcae_batches[ar_bucket]["image_ids"].append(imgid)
				del ar_bucket

				# Check queue and batch-process if we gathered enough samples
				ar_buckets = list(dcae_batches.keys())
				for ar_bucket in ar_buckets:
					target_split = f"{split}_{ar_bucket}"     # name of split the images of this batch belong to

					if (
						# batch is full -> process
						(len(dcae_batches[ar_bucket]["labels"]) >= dcae_batch_size)
						or 
						# batch is not full but we reached end of dataset -> process
						(idx == indices[-1] and len(dcae_batches[ar_bucket]["labels"]) > 0)
					):
						latents = process_dcae_batch(dcae_batches[ar_bucket], dcae)
						dataset_list.extend(latents)

						# empty the dcae batch we just processed
						dcae_batches[ar_bucket]={"labels": [], "images": [], "image_ids": []}

						# debug: save the images of the first processed batch
						if not zerobatch_saved:
							make_grid(
								[
									pil_add_text(
										latent_to_PIL(item["latent"].to(dcae.dtype).to(dcae.device), dcae),
										item["label"],
										stroke_width=2,
										position=(0,0),
										font_size=10,
									)
									for item in latents[:64]
								], 4, 8
							).save(f"zerobatch_rank{rank}.png")
							zerobatch_saved=True

				# upload to HF if we gathered more than upload_every OR reached the end 
				if (
					# processed enough -> upload
					(len(dataset_list) >= upload_every)
					or 
					# reached end of dataset -> upload
					(idx == indices[-1] and len(dataset_list) > 0)
				):
					# try to not upload at exactly the same time to the hub
					lock_file = acquire_lock()
					print(f"Lock acquired by rank {rank}")
					Dataset.from_list(dataset_list).push_to_hub(
						hf_dataset, 
						revision="main",
						commit_message=f"{split}_worker_{rank}.part_{parts_uploaded}", 
						split=f"{split}_worker_{rank}.part_{parts_uploaded}", 
						num_shards=1
					)
					release_lock(lock_file)
					print(f"Lock released by rank {rank}")

					parts_uploaded+=1
					samples_uploaded += len(dataset_list)
					print(f"Rank {rank} uploaded",len(dataset_list), "samples of split", split, "part", parts_uploaded)
					dataset_list=[]   
					time.sleep(1) # don't enter loop immediately, let the others play too

				pbar.update(1)

		pbar.close()
		print(f"split {split}, rank{rank}: total samples uploaded:", samples_uploaded)

if __name__ == '__main__':
	ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

	if ddp:
		dist.init_process_group(backend='nccl')
		rank = dist.get_rank()
		world_size = dist.get_world_size()
		torch.cuda.set_device(rank)
	else:
		rank = 0
		world_size = 1
	is_master = rank == 0  

	process(rank, is_master, world_size)

	if ddp:
		dist.destroy_process_group()
