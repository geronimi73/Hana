import torch.distributed as dist, torch, random, time, os, json, pdb, asyncio
import shutil, threading, queue
import datasets
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk
from diffusers import AutoencoderDC
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from time import sleep

from huggingface_hub import create_repo

from utils import make_grid, PIL_to_latent, latent_to_PIL, pil_add_text
from utils_preprocess import (
    pad, 
    acquire_lock,
    release_lock,
    resizeToClosestMultOf,
    hf_list_files,
    download_with_retry,
    list_json_files,
    load_and_resize_image,
    extract_tar,
    apply_fn_parallel,
)

test_run = False
dtype = torch.bfloat16
device = "cuda:3"

ds_source_repo = "gmongaras/CC12M_and_Imagenet21K_Recap_Highqual"
ds_target_repo = "g-ronimo/CC12M_IN21K-256px_dc-ae-f32c32-sana-1.0"
resizeTo = 256
download_batch_size = 128  
dcae_batch_size = 32  # don't set this too high to keep the GPU busy
dcae_f = 32  
ds_tmp_dir="./dataset"
zerobatch_saved = False # ignore this

# maybe torch.dist?
rank = 0

# Queue to hold downloaded files waiting to be processed
extractor_queue = queue.Queue(maxsize=4)  # Limiting queue size to control memory usage
processor_queue = queue.Queue(maxsize=4)  # Limiting queue size to control memory usage

print("Creating target repo")
create_repo(ds_target_repo, private=True, exist_ok=True, repo_type="dataset")

print("Loading models","rank", rank)
model = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
dcae = AutoencoderDC.from_pretrained(model, subfolder="vae", torch_dtype=dtype).to(device)

print(f"Loading dataset {ds_source_repo}","rank", rank)
all_files = hf_list_files(ds_source_repo, "data/*.parquet")
all_files = [os.path.basename(file) for file in all_files]
processed_files_on_hub = hf_list_files(ds_target_repo+"/data", pattern="*.parquet")

unprocessed_files = []
for fn in all_files:
    is_processed = any(
        [
            # - is replaced with _ during uploading!
            f"/{fn.split('.')[0]}_".replace("-","_") in fn_proc 
            for fn_proc in processed_files_on_hub
        ]
    )
    if not is_processed: 
        unprocessed_files.append("data/" + fn)

print(f"All files: {len(all_files)}")
print(f"Processed files: {len(processed_files_on_hub)}")
print(f"Unprocessed files: {len(unprocessed_files)}")

# numbers check out?
assert len(all_files) - len(processed_files_on_hub) == len(unprocessed_files)

def main():
    
    # Start the downloader thread
    download_thread = threading.Thread(target=downloader_thread)
    download_thread.start()
    
    # Start the processor thread
    process_thread = threading.Thread(target=processor_thread)
    process_thread.start()
    
    # Wait for downloader to complete
    download_thread.join()
    print("Downloading done, waiting for extract")
            
    process_thread.join()
    
    print("All downloads and processing complete")

def downloader_thread():
    for unprocessed_file in tqdm(unprocessed_files, desc="Downloader thread"):
        tar_file = download_file(unprocessed_file)
        processor_queue.put(tar_file)
        if test_run: 
            processor_queue.put(None)
            break

def download_file(tar_file):
    tar_file_local = ds_tmp_dir + "/" + tar_file

    if os.path.exists(tar_file_local):
        print(f"Skipping download of {tar_file} from repo {ds_source_repo}: already exists")
    else:
        print(f"Downloading f{tar_file} from repo {ds_source_repo}")
        tar_file_local = download_with_retry(
            repo_id=ds_source_repo, 
            filename=tar_file, 
            repo_type="dataset", 
            local_dir=ds_tmp_dir
        )

    return tar_file_local

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

def process_parquet(tar_file):
    global zerobatch_saved
    dataset_list = []   # list of dicts per AR bucket, each containing list of {label=.., latent=..}
    dcae_batches = {}   # dict of batches, grouped by latent shape as dict key
    tar_file_stem = Path(tar_file).stem

    ds_images = load_dataset("parquet", data_files=tar_file, num_proc=6)["train"]
    ds_images = ds_images.cast_column("image", datasets.Image())
    if test_run:
        ds_images = ds_images.select(range(50))

    for idx, item in enumerate(
        tqdm(ds_images, f"Generating latents for {tar_file}, rank {rank}")
    ):
        img, imgid, label = item["image"], item["id"], item["recaption_short"]

        # resize
        img = resizeToClosestMultOf(
            img, 
            resizeTo=resizeTo, 
            denominator=dcae_f, 
            maxDim=2*resizeTo,
        ) 

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
                (idx == len(ds_images)-1 and len(dcae_batches[ar_bucket]["labels"]) > 0)
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
    # print(f"Lock acquired by rank {rank}")
    # TODO: IMPLEMENT RETRIES
    max_retries = 200
    retry_delay = 60
    for attempt in range(max_retries):
        try:
            dataset_hf.push_to_hub(
                ds_target_repo, 
                revision="main",
                commit_message=f"{tar_file_stem}_worker_{rank}", 
                split=f"{tar_file_stem}_worker_{rank}".replace("-","_"), 
                num_shards=1
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Upload failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                print(e)
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Skipping file: {tar_file_stem}")
                return None

    release_lock(lock_file)
    
    # delete downloaded parquet in the end
    os.remove(tar_file)
    # print(f"Lock released by rank {rank}")

    # print(f"Rank {rank} uploaded",len(dataset_list), "samples of file", tar_file)
    # time.sleep(2) # don't enter loop immediately, let the others play too

def processor_thread():
    while True:
        tar_file = processor_queue.get()
        
        # Check for the sentinel value
        if tar_file is None:
            processor_queue.task_done()
            break
        
        process_parquet(tar_file)
        processor_queue.task_done()

if __name__ == "__main__":
    main()