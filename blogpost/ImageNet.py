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
    SanaDiTS,
    SanaDiTB,
    FMNIST_LABEL_TO_DESC
)

lr = 5e-4
bs = 1024
epochs = 10
eval_prompts = [
    "a collection of comic books on a table",
    "a green plant with a green stem",
    "an airplane in the sky",
    "two fighter jets on the red sky",
    "a blonde girl",
    "a red car",
    "a blue car",
    "a cheeseburger on a white plate", 
    "a bunch of bananas on a wooden table", 
    "a white tea pot on a wooden table", 
    "an erupting volcano with lava pouring out",
    "a european castle on a mountain",
    "a red train in the mountains",
    "a photo of the eiffel tower in the dessert",
]

te_repo = "answerdotai/ModernBERT-base"
sana_repo = "Efficient-Large-Model/Sana_600M_1024px_diffusers"
dataset_repo ="g-ronimo/IN1k-128-latents_dc-ae-f32c32-sana-1.0"

set_seed(42)

# Choose device and dtype
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# Load the text encoder and AE
tokenizer = AutoTokenizer.from_pretrained(te_repo, torch_dtype=dtype)
text_encoder = AutoModel.from_pretrained(te_repo, torch_dtype=dtype).to(device)
dcae = AutoencoderDC.from_pretrained(sana_repo, subfolder="vae", torch_dtype=dtype).to(device)

# Initialize the DiT, DiT-S with 12 layers, hidden size 384
transformer = SanaDiTB().to(dtype).to(device)

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
    # drop 10% of the labels
    labels = [ i["label"] if random() > 0.1 else "" for i in items ]

    latents = torch.Tensor([i["latent"] for i in items])
    B, num_aug, _, _, _ = latents.shape
    aug_idx = 0
    batch_idx = torch.arange(B)
    latents = latents[batch_idx, aug_idx] 

    # latents = torch.cat(
    #     [torch.Tensor(i["latent"][0])[None] for i in items]
    # )

    return labels, latents

ds = load_dataset(dataset_repo)
dataloader_train = DataLoader(ds["train"], batch_size=bs, collate_fn=collate, num_workers=4, prefetch_factor=2)
dataloader_eval = DataLoader(ds["test"], batch_size=256, collate_fn=collate)

# Optimizer
optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)

# Eval: images
def eval_images(seed = None):
    return transforms.ToPILImage()(
        make_grid(
            [
                transforms.ToTensor()(
                    generate(
                        prompt,
                        latent_dim = [1, 32, 4, 4],
                        num_steps = 10,
                        guidance_scale = guidance_scale,
                        seed = seed
                    )
                ) 
                for prompt in tqdm(eval_prompts, "eval_images")
                for guidance_scale in [1, 2, 7]
            ], 
            nrow=9
        )
    )

# Eval: clip score
def eval_clipscore():
    images = [
        generate(
            prompt,
            # 4x4 latents -> 128x128px images
            latent_dim=[1, 32, 4, 4],
            guidance_scale=7,
            num_steps=10,
        )
        for prompt in tqdm(eval_prompts, "eval_clipscore")
    ]

    return pil_clipscore(images, eval_prompts)

# Eval: loss
def eval_loss():
    losses = []

    for batch_num, (labels, latents) in tqdm(enumerate(dataloader_eval), "eval_loss"):
        # Encode prompts
        prompts_emb, prompts_atnmask = encode_prompt(labels, tokenizer, text_encoder, max_length=10)

        latents *= dcae.config["scaling_factor"]
        latents = latents.to(dtype).to(device)
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
    project="IN128px", 
    name=f"{transformer_params:.1f}M_BS-{bs}_LR-{lr}_{epochs}-epochs"
).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".json"))

# TRAIN
step = 0 
for e in range(epochs):
    for labels, latents in dataloader_train:
        step += 1
        epoch = step/len(dataloader_train)

        # Encode prompts
        prompts_emb, prompts_atnmask = encode_prompt(labels, tokenizer, text_encoder, max_length=10)

        # Scale latent and add random amount of noise
        latents = latents.to(dtype).to(device)
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
            eval_clipscore()

        if step % 300 == 0:
            wandb.log({"step": step, "epoch": epoch, "eval_images": wandb.Image(eval_images(seed=42))})

    # eval after each epoch
    el = eval_loss()
    print(f"eval step {step} epoch {epoch:.2f} loss: {el:.2f}")
    wandb.log({"step": step, "epoch": epoch, "loss_eval": el, "eval_clipscore": eval_clipscore()})

    # save after each epoch
    transformer.save_pretrained(f"IN96px_e{e}")

# log a big 10x10 gallery 
wandb.log({"final_gallery": wandb.Image(eval_images(num_rows=10))})

wandb.finish()



