import torch
import torch.distributed as dist
import json, pickle, glob, re, os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from PIL import Image
from utils import load_imagenet_labels
from torchvision import transforms as T
from tqdm import tqdm

from PIL import PngImagePlugin
# otherwise might lead to Decompressed Data Too Large for some images
LARGE_ENOUGH_NUMBER = 10
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

from utils_captioning import batch_caption_qwenvlm

# Save checkpoint to disk
def save_checkpoint(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved to {filename}")

# Load checkpoint from disk
def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"Checkpoint loaded from {filename}")
    return data

def find_latest_cp(rank, split, cp_template):
    # Regex for batch num
    cp_regex = cp_template.format(rank=rank, split=split, batch_num=r"(\d+)")
    cp_regex = re.compile(cp_regex)
    
    # Pattern for matching all files of given rank and split
    pattern = cp_template.format(rank=rank, split=split, batch_num="*")
    
    highest_batch_num = -1
    for file in glob.glob(pattern):
        batch_num = int(cp_regex.match(file).group(1))
        if batch_num>highest_batch_num: highest_batch_num=batch_num

    return highest_batch_num

def process(rank, is_master, world_size):
	print(f"Hello from rank {rank}")

	## Parameters 
	model_repo = "Qwen/Qwen2.5-VL-7B-Instruct"
	ds_repo = "visual-layer/imagenet-1k-vl-enriched"
	ds_target = "g-ronimo/imagenet-1k-vl-enriched_qwen2.4vlm"
	in_labels = load_imagenet_labels()
	prompt_template = """This is an image of a {class_name}. 
Please write a short caption based on the image content. 
Describe text if there is any.
Describe the main objects' colors.
Keep the caption short, precise and in simple english.
Output the caption only."""

	batch_size = 12
	cp_template = "checkpoints/checkpoint_split-{split}_rank{rank}_batch-{batch_num}.pkl"
	cp_every = 500
	# start from latest available checkpoint?
	start_from_cp = False

	# Create cp dir
	directory = os.path.dirname(cp_template)
	if not os.path.exists(directory): os.makedirs(directory)	

	# Load model 
	dtype = torch.bfloat16
	device = "cuda" 

	# Load dataset
	ds = load_dataset(ds_repo, cache_dir="~/ssd-2TB/hf_cache")

	# Distributed data loader without shuffling
	def collate_fn(items):
		return [
			[i["image_id"] for i in items],
			[T.Resize(256)(i["image"]) for i in items],
			[in_labels[i["label"]] for i in items],
			[i["caption_enriched"] for i in items],
		]

	# Process your data
	for split in list(ds.keys()):
		sampler = DistributedSampler(ds[split], shuffle=False)
		dataloader = DataLoader(
			ds[split], 
			sampler=sampler, 
			collate_fn=collate_fn, 
			batch_size=batch_size,
			num_workers=2, 
			prefetch_factor=4
		)

		print(f"dataset size on rank {rank}: {len(dataloader)}")

		if start_from_cp:
			cp_start = find_latest_cp(rank, split, cp_template)
			if cp_start != -1:
				print(f"Rank {rank} loading cp {cp_start} for split {split}")
				moondream_captions = load_checkpoint(
					cp_template.format(split=split, rank=rank, batch_num=cp_start)
				)
			else:
				moondream_captions = dict(image_id=[], caption=[])
		else:
			moondream_captions = dict(image_id=[], caption=[])

		for batch_num, (image_ids, images, labels, captions) in tqdm(
			enumerate(dataloader), 
			desc=f"rank {rank}, split {split}",
			total=len(dataloader)
		):
			batch_num+=1

			## Two possible ways of skipping over already processed samples if starting from a checkpoint
			## 1
			# if start_from_cp and batch_num<cp_start: continue
			## 2
			is_processed = [image_id in moondream_captions["image_id"] for image_id in image_ids]
			if all(is_processed): continue

			prompts = [
				prompt_template.format(
					class_name=labels[i],
					caption=captions[i],
				)
				for i in range(len(image_ids))
			]

			captions_md = batch_caption_qwenvlm(
				images=images,
				prompts=prompts,
				repo=model_repo,
			)

			moondream_captions["image_id"].extend(image_ids)
			moondream_captions["caption"].extend(captions_md)

			if batch_num % cp_every == 0:
				save_checkpoint(
					moondream_captions, 
					cp_template.format(split=split, rank=rank, batch_num=batch_num)
				)

		# Upload to HF
		ds_md = Dataset.from_dict(moondream_captions)
		ds_md.push_to_hub(ds_target, 
			split=f"{split}_rank{rank}", 
			private=True, num_shards=1
		)

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
