import torch
import torch.distributed as dist
import torchvision.transforms as T
import numpy as np
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from types import SimpleNamespace
from diffusers import AutoencoderDC
from utils import make_grid, PIL_to_latent, latent_to_PIL, dcae_scalingf
from PIL import PngImagePlugin
# otherwise might lead to Decompressed Data Too Large for some images
LARGE_ENOUGH_NUMBER = 10
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

def get_average_color(img):
	img_array = np.array(img)
	return tuple(map(int, img_array.mean(axis=(0,1))))

def resize_and_augment(img, resizeTo, do_augment=False):
	def noop(x): return x
	img = img.convert('RGB') if img.mode != "RGB" else img
	transform = T.Compose([
		T.RandomHorizontalFlip(p=0.5 if do_augment else 0),
		T.RandomRotation([-5,5],expand=False, fill=get_average_color(img)) if do_augment else noop,
		T.Resize(resizeTo, antialias=True),
		T.CenterCrop(resizeTo),
	])
	return transform(img)


def process(rank, is_master, world_size):
	config = SimpleNamespace(
		test_run = False,
		dtype=torch.bfloat16,
		bs=64,
		resizeTo=128,
		augs_per_img=4,
		ds_repo="visual-layer/imagenet-1k-vl-enriched",
		col_label="caption_enriched",
		col_image="image",
		dcae_repo=dict(
			pretrained_model_name_or_path="Efficient-Large-Model/Sana_600M_1024px_diffusers",
			subfolder="vae"
		),
		upload_to="g-ronimo/IN1k-128-latents_dc-ae-f32c32-sana-1.0",
		upload_every=400.  # number of batches
	)

	# Load DCAE
	dcae = AutoencoderDC.from_pretrained(**config.dcae_repo, torch_dtype=config.dtype).cuda()

	# Load dataset
	ds = load_dataset(config.ds_repo, cache_dir="~/ssd-2TB/hf_cache")
	splits = list(ds.keys())

	# Setup distributed data loader without shuffling
	def collate_fn(items):
		labels = [item[config.col_label] for item in items]
		images = [
			[
				# image 0 is always the original
				resize_and_augment(
					item[config.col_image], 
					resizeTo=config.resizeTo,
					do_augment=True if i>0 else False
				) 
				for i in range(config.augs_per_img + 1)
			]
			for item in items
		]
		return labels, images

	# Processing loop
	for split in splits:
		sampler = DistributedSampler(ds[split], shuffle=False)
		dataloader = DataLoader(
			ds[split], 
			sampler=sampler, 
			collate_fn=collate_fn, 
			batch_size=config.bs,
			num_workers=2,
			prefetch_factor=10,

		)
		dataset_latents = []
		dataset_part = 0
		num_batches = len(dataloader)
		if is_master:
			print(f"DEBUG {len(dataloader)} batches, batch size {config.bs}, uploading every {config.upload_every}")
			print(f"DEBUG Resulting number of parts: {len(dataloader)/config.upload_every}")

		# Process your data
		for batch_no, (labels, images) in tqdm(
			enumerate(dataloader), 
			desc=f"rank {rank}, split {split}",
			total=len(dataloader),
		):
			# images: list of lists 

			# inspect first batch images
			if batch_no == 0:
				batch_images = [
					make_grid(augmentations)
					for i, augmentations in enumerate(images)
				]
				batch_gallery = make_grid(batch_images, len(images), 1)
				batch_gallery.save(f"zerobatch_rank-{rank}_images.png")

			# flatten images for encoding
			images_flat = [i for augmentations in images for i in augmentations ]

			# encode. it's a tensor. not flat but shape [B * (augs_per_img+1), C, W, H]
			latents_flat = PIL_to_latent(images_flat, dcae).cpu()  

			# inspect first batch latents
			if batch_no == 0:
				print("Latents shape:", latents_flat.shape)

				latents_flat = latents_flat.cuda()
				batch_images = []
				for img_idx in range(len(images)):
					batch_images.append(
						make_grid([
								latent_to_PIL(latents_flat[None, latent_idx], dcae)
								for latent_idx in range(
									img_idx*(config.augs_per_img+1), 
									(img_idx+1)*(config.augs_per_img+1),
								)
						])
					)
				batch_gallery = make_grid(batch_images, len(images), 1)
				batch_gallery.save(f"zerobatch_rank-{rank}_latents.png")

			# Add latents to list 
			for img_idx, label in enumerate(labels):
				idx_start = img_idx*(config.augs_per_img+1)
				idx_end = (img_idx+1)*(config.augs_per_img+1)

				latents = latents_flat[idx_start:idx_end]
				dataset_latents.append({
					"label": label,
					"latent": latents 
				})

			# Upload every x batches
			if (batch_no+1) % config.upload_every == 0 or (batch_no+1) % num_batches == 0:
				print(f"Rank {rank}, uploading split {split}/part {dataset_part} to {config.upload_to}")
				Dataset.from_list(dataset_latents).push_to_hub(
					config.upload_to, 
					split=f"{split}.{rank}.{dataset_part}", 
					num_shards=1
				)
				dataset_part+=1
				dataset_latents=[]


if __name__ == '__main__':
	dist.init_process_group(backend='nccl')

	seed = 42
	torch.manual_seed(seed)

	world_size = dist.get_world_size()
	rank = dist.get_rank()
	is_master = rank == 0  
	torch.cuda.set_device(rank)

	process(rank, is_master, world_size)

	dist.destroy_process_group()
