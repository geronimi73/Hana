import torch, torch.nn.functional as F, random, wandb, time
import torchvision.transforms as T
from torchvision import transforms
from diffusers import AutoencoderDC, SanaTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import AutoModel, AutoTokenizer, set_seed
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
from torch.utils.data import DataLoader
from functools import partial
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from math import ceil

from utils import load_imagenet_labels, latent_to_PIL, pil_add_text, make_grid, encode_prompt, dcae_scalingf, pil_clipscore, cifar10_labels, free_memory, mnist_labels

def load_models(text_encoder, transformer_config, ae, dtype, device):
	transformer = SanaTransformer2DModel.from_config(transformer_config).to(device).to(dtype)
	te = AutoModel.from_pretrained(text_encoder, torch_dtype=dtype).to(device)
	tok = AutoTokenizer.from_pretrained(text_encoder, torch_dtype=dtype)
	dcae = AutoencoderDC.from_pretrained(ae, subfolder="vae", torch_dtype=dtype).to(device)

	if is_master:
		print(f"Transformer parameters: {sum(p.numel() for p in transformer.parameters()) / 1e6:.2f}M")
		print(f"DCAE parameters: {sum(p.numel() for p in dcae.parameters()) / 1e6:.2f}M")

	return transformer, te, tok, dcae

def load_data(repo_name, col_latent = "latent", col_label = "label"):
	if is_master:
		print(f"loading dataset {repo_name}")

	ds = load_dataset(repo_name)
	# ImageNet
	del ds["test"]

	splits = [k for k in ds]
	# latent_shape = torch.Tensor(ds[splits[0]][0][col_latent]).shape
	# ImageNet
	latent_shape = torch.Tensor(ds[splits[0]][0][col_latent])[None,].shape
	features = ds[splits[0]].features

	assert len(splits)==2
	assert len(latent_shape)==4, f"Latent shape not 4! {latent_shape}"
	assert col_latent in features and col_label in features

	if is_master:
		for i, split in enumerate(splits): print(f"split #{i} {split}: {len(ds[split])} samples, features: {[k for k in ds[split].features]}")
		print(f"latent shape {latent_shape}")

	return ds, splits, latent_shape

def collate_(items, labels_encoded, col_latent = "latent", col_label = "label"):
	assert col_latent in items[0] and col_label in items[0]
	labels = [i[col_label] for i in items]
	# latents = torch.cat([torch.Tensor(i[col_latent]) for i in items]).to(dtype).to(device)
	# ImageNet dataset is [32, 8, 8] instead of [1, 32, 8, 8] => stack!	
	latents = torch.stack([torch.Tensor(i[col_latent]) for i in items]).to(dtype).to(device)
	prompts_encoded = torch.cat([labels_encoded[label][0] for label in labels])
	prompts_atnmask = torch.cat([labels_encoded[label][1] for label in labels])

	return labels, latents, prompts_encoded, prompts_atnmask

def get_dataloaders(ds, splits, bs):
	batch_sizes = [bs, bs]    # reduce eval_batch size
	dataloaders = []
	for i, split in enumerate(splits):
		batch_size = batch_sizes[i]
		if is_master:
			print(f"Assuming split #{i} \"{split}\" is {'train' if i==0 else 'test'} split, testing batch size {batch_size}")
		sampler = DistributedSampler(
			ds[split], 
			shuffle = True,
			seed = seed
		)
		dataloader = DataLoader(
			dataset=ds[split], 
			batch_size=batch_size, 
			shuffle=False, 
			sampler = sampler,
			collate_fn=collate,
			# pin_memory = True,
		)
		b = next(iter(dataloader))
		for i, col in enumerate(b):
			coltype = type(col)
			collength = len(col) if coltype==list else col.shape
			if is_master:
				print(f" col {i} {coltype.__name__} {collength}")
		dataloaders.append(dataloader)
	return dataloaders

def get_timesteps(num_steps):
	dt = 1.0 / num_steps
	timesteps = [int(i/num_steps*1000) for i in range(num_steps, 0, -1)]
	return dt, timesteps

def generate(prompt, tokenizer, text_encoder, latent_dim=None, num_steps=100, latent_seed=42):
	assert latent_dim is not None

	dt, timesteps = get_timesteps(num_steps)
	prompt_encoded, prompt_atnmask = encode_prompt(str(prompt), tokenizer, text_encoder)
	latent = torch.randn(latent_dim, generator = torch.manual_seed(latent_seed)).to(dtype).to(device)
	for t in timesteps:
		t = torch.Tensor([t]).to(dtype).to(device)
		with torch.no_grad():
			noise_pred = transformer(latent, encoder_hidden_states=prompt_encoded, timestep=t, encoder_attention_mask=prompt_atnmask, return_dict=False)[0]
		latent = latent - dt * noise_pred

	return latent_to_PIL(latent / dcae_scalingf, dcae)

def add_random_noise(latents, timesteps=1000):
	noise = torch.randn_like(latents)
	t = torch.randint(1, timesteps + 1, (latents.size(0),)).to(device)
	tperc = t.view([latents.size(0), *([1] * len(latents.shape[1:]))])/timesteps
	latents_noisy = (1 - tperc) * latents + tperc * noise # (1-noise_level) * latent + noise_level * noise

	return latents_noisy, noise, t

def eval_loss(dataloader_eval, timesteps=1000, testing=False):
	losses = []

	for batch_num, (labels, latents, prompts_encoded, prompts_atnmask) in tqdm(enumerate(dataloader_eval), "eval_loss"):
		latents = latents * dcae_scalingf
		latents_noisy, noise, t = add_random_noise(latents, timesteps)
		with torch.no_grad():
			noise_pred = transformer(latents_noisy.to(dtype), prompts_encoded, t, prompts_atnmask).sample

		loss = F.mse_loss(noise_pred, noise - latents)
		losses.append(loss.item())  
		if testing: break
	return sum(losses)/len(losses)

def eval_images(prompts):
    images = [
        generate(p, tokenizer, text_encoder, latent_dim=latent_shape, num_steps=10) 
        for p in tqdm(prompts, "eval_images")
    ]    
    images_labeled = [pil_add_text(images[i], prompt) for i, prompt in enumerate(prompts)]

    return images, images_labeled

if __name__ == '__main__':
	dist.init_process_group(backend='nccl')

	# Seed first 
	seed = 42
	set_seed(seed)

	# DDP
	is_master = dist.get_rank() == 0  
	world_size = dist.get_world_size()
	local_rank = dist.get_rank()
	torch.cuda.set_device(local_rank)

	# Training params
	log_wandb = True
	dtype = torch.bfloat16
	device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"
	lr = 5e-4
	bs = 256
	epochs = 1000
	timesteps_training = 1000
	# steps_log, steps_eval = 20, 300
	steps_log, steps_eval = 20, 1000
	wandb_project = "Hana"

	# Load all the models
	transformer, text_encoder, tokenizer, dcae = load_models(
		text_encoder = "answerdotai/ModernBERT-base",
		transformer_config = "transformer_Sana-7L-MBERT_config.json",
		# transformer_config = "transformer_Sana-xsmall.json",
		ae = "Efficient-Large-Model/Sana_600M_1024px_diffusers",
		dtype = dtype,
		device = device
	)
	transformer = DistributedDataParallel(transformer, device_ids=[local_rank])

	# Load dataset
	ds, ds_splits, latent_shape = load_data("g-ronimo/Imagenet-256-latents_dc-ae-f32c32-sana-1.0")
	labels = load_imagenet_labels()
	labels = {k:v for k,v in enumerate(labels)}
	eval_labels = labels
	# CHECK THIS LINE WHENEVER YOU CHANGE THE DATASET!
	collate = partial(collate_, labels_encoded = {k: encode_prompt(str(labels[k]), tokenizer, text_encoder) for k in labels})
	dataloader_train, dataloader_eval = get_dataloaders(ds, ds_splits, bs)
	steps_epoch = len(dataloader_train)

	# Run some tests on master
	if is_master:
		# print("Labels:")
		# for k in labels: print(f"{k}: {labels[k]}")

		print("Inspecting first batch")
		def inspect_first_batch():
			labels, latents, prompts_encoded, prompts_atnmask = next(iter(dataloader_train))
			print(labels, [eval_labels[l] for l in labels])
			print("latents shape of batch", latents.shape)
			make_grid(latent_to_PIL(latents, dcae), 10, 10).save("test_first_batch.png")
		# OOM because latent_to_PIL on 256 samples is too many 
		# inspect_first_batch()

		print("Testing eval loss")
		print(eval_loss(dataloader_eval, testing=True))

		print("Testing eval images and clip score")
		prompts = [eval_labels[k] for k in eval_labels]
		prompts = prompts[:20]
		images, images_labeled = eval_images(prompts)
		make_grid(images_labeled, ceil(len(images)/5), 5).save("test_eval_images.png")
		print(pil_clipscore(images, prompts))

		print(f"steps per epoch: {steps_epoch}")

		model_size = sum(p.numel() for p in transformer.parameters())
		wandb_run = f"{model_size / 1e6:.2f}M_CIFAR-10_LR-{lr}_BS-{bs}_TS-{timesteps_training}_DDP-{world_size}x3090"
		del prompts

	optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)
	del labels
	free_memory()

	# Setup wandb
	if log_wandb and is_master: 
		if wandb.run is not None: wandb.finish()
		wandb.init(project=wandb_project, name=wandb_run).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".json"))

	# TRAIN!
	transformer.train()
	step = 0
	last_step_time = time.time()

	for epoch in range(epochs):
		dataloader_train.sampler.set_epoch(epoch)

		for labels, latents, prompts_encoded, prompts_atnmask in dataloader_train:
			latents = latents * dcae_scalingf
			latents_noisy, noise, t = add_random_noise(latents)
			noise_pred = transformer(latents_noisy.to(dtype), prompts_encoded, t, prompts_atnmask).sample

			optimizer.zero_grad()    
			loss = F.mse_loss(noise_pred, noise - latents)
			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
			optimizer.step()

			if is_master and step>0 and step % steps_log == 0:
				loss_train = loss.item()
				step_time = (time.time() - last_step_time) / steps_log * 1000
				sample_tp = (bs * world_size * steps_log) / (time.time() - last_step_time)
				print(f"step {step}, epoch: {step / steps_epoch:.4f}, train loss: {loss_train:.4f}, grad_norm: {grad_norm:.2f}, {step_time:.2f}ms/step, {sample_tp:.2f}samples/sec")
				if log_wandb: wandb.log({"loss_train": loss_train, "grad_norm": grad_norm, "step_time": step_time, "step": step, "sample_tp": sample_tp, "sample_count": step * bs * world_size, "epoch": step / steps_epoch})
				last_step_time = time.time()

			if is_master and step>0 and step % steps_eval == 0:
				transformer.eval()
				loss_eval = eval_loss(dataloader_eval)
				prompts = [eval_labels[k] for k in eval_labels][:20]
				val_images, val_images_labeled = eval_images(prompts)
				val_images_labeled = make_grid(val_images_labeled, ceil(len(val_images_labeled)/5), 5)
				clipscore = pil_clipscore(val_images, prompts)
				print(f"step {step}, eval loss: {loss_eval:.4f}, clipscore: {clipscore:.2f}")
				if log_wandb: wandb.log({"loss_eval": loss_eval, "clipscore": clipscore, "images_eval": wandb.Image(val_images_labeled), "step": step, "sample_count": step * bs * world_size, "epoch": step / steps_epoch})
				free_memory()

				transformer.train()
			step += 1

	if is_master and log_wandb: wandb.finish()
	dist.destroy_process_group()


