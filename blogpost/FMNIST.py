import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from transformers import set_seed, AutoModel, AutoTokenizer
from diffusers import AutoencoderDC, SanaTransformer2DModel
from datasets import load_dataset
from functools import partial
from tqdm import tqdm
from random import random

from utils import (
    latent_to_PIL, 
    PIL_to_latent, 
    generate, 
    encode_prompt, 
    add_random_noise,
    pil_clipscore,
    FMNIST_LABEL_TO_DESC
)

lr = 1e-3
bs = 512
epochs = 10

te_repo = "answerdotai/ModernBERT-base"
sana_repo = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
dataset_repo ="g-ronimo/FMNIST-latents-64_dc-ae-f32c32-sana-1.0"

set_seed(42)

# Choose device and dtype
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# Load the text encoder and AE
tokenizer = AutoTokenizer.from_pretrained(te_repo, torch_dtype=dtype)
text_encoder = AutoModel.from_pretrained(te_repo, torch_dtype=dtype).to(device)
dcae = AutoencoderDC.from_pretrained(sana_repo, subfolder="vae", torch_dtype=dtype).to(device)

# Initialize the DiT, DiT-S with 12 layers, hidden size 384
config = SanaTransformer2DModel.load_config(sana_repo, subfolder="transformer")
config["num_layers"] = 12
config["caption_channels"] = 768
config["num_attention_heads"] = 6
config["attention_head_dim"] = 64
config["cross_attention_dim"] = 384
config["num_cross_attention_heads"] = 6

transformer = SanaTransformer2DModel.from_config(config).to(dtype).to(device)

transformer_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6
print(f"Transformer parameters: {transformer_params:.2f}M")

# Fix generate with pipeline components
generate = partial(
    generate, 
    transformer=transformer, 
    tokenizer=tokenizer, 
    text_encoder=text_encoder,
    dcae=dcae
)

# Prepare dataloader
def collate(items):
    labels = [
        FMNIST_LABEL_TO_DESC[i["label"]]
        for i in items
    ]
    prompts_emb, prompts_atnmask = encode_prompt(labels, tokenizer, text_encoder, max_length=5)

    latents = torch.cat(
        [torch.Tensor(i["latent"]) for i in items]
    ).to(dtype).to(device)

    return labels, prompts_emb, prompts_atnmask, latents

ds = load_dataset(dataset_repo)
dataloader_train = DataLoader(ds["train"], batch_size=bs, collate_fn=collate)
dataloader_eval = DataLoader(ds["test"], batch_size=256, collate_fn=collate)

# Optimizer
optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)

# Eval: images
def eval_images(num_rows=1, seed = None):
    num_images = num_rows * 10
    return transforms.ToPILImage()(
        make_grid(
            [
                transforms.ToTensor()(
                    generate(
                        FMNIST_LABEL_TO_DESC[i % 10],
                        latent_dim = [1, 32, 2, 2],
                        num_steps = 10,
                        seed = seed
                    )
                ) 
                for i in tqdm(range(num_images), "eval_images")
            ], 
            nrow=10
        )
    )

# Eval: clip score
def eval_clipscore(num_images = 100):
    prompts = [FMNIST_LABEL_TO_DESC[i % 10] for i in range(100)]
    images = [
        generate(
            prompt,
            latent_dim = [1, 32, 2, 2],
            num_steps = 10,
        )
        for prompt in tqdm(prompts, "eval_clipscore")
    ]

    return pil_clipscore(images, prompts)

# Eval: loss
def eval_loss():
    losses = []

    for batch_num, (labels, prompts_emb, prompts_atnmask, latents) in tqdm(enumerate(dataloader_eval), "eval_loss"):
        latents = latents * dcae.config["scaling_factor"]
        latents_noisy, timestep, noise = add_random_noise(latents)
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states = latents_noisy.to(dtype), 
                encoder_hidden_states = prompts_emb, 
                encoder_attention_mask = prompts_atnmask,
                timestep = timestep, 
            ).sample
    
        loss = F.mse_loss(noise_pred, noise - latents)
        losses.append(loss.item())  
    return sum(losses)/len(losses)

# Setup logging 
wandb.init(
    project="FMNIST", 
    name=f"{transformer_params:.1f}M_BS-{bs}_LR-{lr}_{epochs}-epochs"
).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".json"))

# TRAIN
step = 0 
for e in range(epochs):
    for labels, prompts_emb, prompts_atnmask, latents in dataloader_train:
        step += 1
        epoch = step/len(dataloader_train)

        # Scale latent and add random amount of noise
        latents *=  dcae.config["scaling_factor"]
        latents_noisy, timestep, noise = add_random_noise(latents)
        
        # Get a noise prediction out of the model
        noise_pred = transformer(
            hidden_states = latents_noisy.to(dtype), 
            encoder_hidden_states = prompts_emb, 
            encoder_attention_mask = prompts_atnmask,
            timestep = timestep, 
        ).sample
        
        loss = F.mse_loss(noise_pred, noise - latents)
        loss.backward()
    
        optimizer.step()
        optimizer.zero_grad()
    
        if step % 10 == 0:
            print(f"step {step} epoch {epoch:.2f} loss: {loss.item()}")
            wandb.log({"step": step, "epoch": epoch, "loss_train": loss.item()})

        if step % 100 == 0:
            wandb.log({"epoch": round(epoch, 1), "step": step, "eval_images": wandb.Image(eval_images(seed=42))})

    # eval after each epoch
    el = eval_loss()
    print(f"eval step {step} epoch {epoch:.2f} loss: {el:.2f}")
    wandb.log({"step": step, "epoch": epoch, "loss_eval": el, "eval_clipscore": eval_clipscore()})

    # save after each epoch
    transformer.save_pretrained(f"FMNIST_e{e}")

wandb.log({"final_gallery": wandb.Image(eval_images(num_rows=10))})

wandb.finish()



