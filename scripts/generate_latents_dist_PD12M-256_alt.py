import torch.distributed as dist, torch, random, time, os, json, pdb, asyncio
import shutil

from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from diffusers import AutoencoderDC
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from time import sleep

from utils import make_grid, PIL_to_latent, latent_to_PIL, pil_add_text
from utils_preprocess import resizeToClosestMultOf, pad, acquire_lock, release_lock, url_to_image, download_all_images
from utils_preprocess import (
    resizeToClosestMultOf,
    hf_list_files,
    download_with_retry,
    list_json_files,
    load_and_resize_image,
    extract_tar,
    apply_fn_parallel,
)

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
	ds_source_repo="Spawning/pd12m-full"
	# ds_source_repo="g-ronimo/test"
	ds_target_repo = "g-ronimo/PD12M-256px_dc-ae-f32c32-sana-1.0"
	resizeTo = 256
	download_batch_size = 128  
	dcae_batch_size = 32  # don't set this too high to keep the GPU busy
	dcae_f = 32  
	ds_tmp_dir="./dataset"
	zerobatch_saved = False # ignore this

	print("Loading models","rank", rank)
	model = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
	dcae = AutoencoderDC.from_pretrained(model, subfolder="vae", torch_dtype=dtype).to(device)

	print(f"Loading dataset {ds_source_repo}","rank", rank)
	all_files = hf_list_files(ds_source_repo)
	all_files = [os.path.basename(file) for file in all_files]
	processed_files_on_hub = hf_list_files(ds_target_repo+"/data", pattern="*.parquet")

	unprocessed_files = []
	for fn in all_files:
		is_processed = any([f"/{fn.split('.')[0]}_" in fn_proc for fn_proc in processed_files_on_hub])
		if not is_processed: unprocessed_files.append(fn)

	print("all files:",len(all_files))
	print("processed files on hub:",len(processed_files_on_hub))
	print("unprocessed files:",len(unprocessed_files))

	print(f"Found {len(unprocessed_files)} unprocessed files in {ds_source_repo}","rank", rank)
	sleep(10)

	# Poor man's distributed sampler
	indices = list(range(len(unprocessed_files)))  # all ranks
	indices = indices[rank : len(indices) : world_size] # subsample for current rank
	unprocessed_files = [unprocessed_files[i] for i in indices]

	for file_num, file in enumerate(unprocessed_files):
		print(f"Processing file {file_num}/{len(unprocessed_files)}","rank", rank)

		if os.path.exists(f"{ds_tmp_dir}/{file}"):
			tar_file = f"{ds_tmp_dir}/{file}"
		else:
			tar_file = download_with_retry(repo_id=ds_source_repo, filename=file, repo_type="dataset", local_dir=ds_tmp_dir)

		# print(f"Extracting {tar_file}")
		image_metas, extract_dir = extract_tar(tar_file, parent_dir=ds_tmp_dir)

		# Read images and resize them immediately
		resize_fn = partial(load_and_resize_image,
			basedir=extract_dir, resizeTo=resizeTo, denominator=dcae_f, maxDim=2*resizeTo
		)
		images = apply_fn_parallel(resize_fn, image_metas, "Resizing images")

		# Clean tmp dir, delete tar
		shutil.rmtree(extract_dir)
		os.remove(tar_file)

		dataset_list = []   # list of dicts per AR bucket, each containing list of {label=.., latent=..}
		dcae_batches = {}   # buffer, collect samples and batch process when full
		tar_file_stem = Path(tar_file).stem

		for idx, image_dict in tqdm(enumerate(images), f"Processing file {file_num}/{len(unprocessed_files)}: {tar_file}, rank {rank}"):
			img, imgid, label = image_dict["img"], image_dict["id"], image_dict["label"]
			
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
				# target_split = f"{tar_file}_{ar_bucket}"     # name of split the images of this batch belong to

				if (
					# batch is full -> process
					(len(dcae_batches[ar_bucket]["labels"]) >= dcae_batch_size)
					or 
					# batch is not full but we reached end of dataset -> process
					(idx == len(images)-1 and len(dcae_batches[ar_bucket]["labels"]) > 0)
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
		dataset_hf = Dataset.from_list(dataset_list)

		# try to not upload at exactly the same time to the hub
		lock_file = acquire_lock()
		print(f"Lock acquired by rank {rank}")
		dataset_hf.push_to_hub(
			ds_target_repo, 
			revision="main",
			commit_message=f"{tar_file_stem}_worker_{rank}", 
			split=f"{tar_file_stem}_worker_{rank}", 
			num_shards=1
		)
		release_lock(lock_file)
		print(f"Lock released by rank {rank}")

		print(f"Rank {rank} uploaded",len(dataset_list), "samples of file", tar_file)
		time.sleep(1) # don't enter loop immediately, let the others play too

		print(f"file {tar_file}, rank{rank}: total samples uploaded:", len(dataset_list))

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
