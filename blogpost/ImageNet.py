import torch
import torch.nn.functional as F
import wandb
import time
from torchvision import transforms
from torchvision.utils import make_grid
from transformers import set_seed, AutoModel, AutoTokenizer
from diffusers import AutoencoderDC, SanaTransformer2DModel
from functools import partial
from tqdm import tqdm

from utils import (
    latent_to_PIL, 
    PIL_to_latent, 
    generate, 
    encode_prompt, 
    add_random_noise,
    pil_clipscore,
    SanaDiTS,
    SanaDiTB,
    StepLogger,
    pil_concat,
    pil_add_text,
    load_IN1k128px,
    load_IN1k256px,
)

lr = 5e-4
bs = 384
epochs = 100
latent_dim = [1, 32, 8, 8]
eval_prompts = [
    "a mountain landscape",
    "a green plant with a brown stem",
    "a tiny bird",
    "an airplane in the sky",
    "a blonde girl",
    "a red car",
    "a cheeseburger on a white plate", 
    "a bunch of bananas on a wooden table", 
    "a white tea pot on a wooden table", 
    "an erupting volcano with lava pouring out",
    "a sunflower wearing sunglasses",
    "a black cat",
    "a dog in the swimming pool",
    "a european castle on a mountain",
    "a red train in the mountains",
]

te_repo = "answerdotai/ModernBERT-base"
sana_repo = "Efficient-Large-Model/Sana_600M_1024px_diffusers"

set_seed(42)

# Choose device and dtype
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# Load the text encoder and AE
tokenizer = AutoTokenizer.from_pretrained(te_repo, torch_dtype=dtype)
text_encoder = AutoModel.from_pretrained(te_repo, torch_dtype=dtype).to(device)
dcae = AutoencoderDC.from_pretrained(sana_repo, subfolder="vae", torch_dtype=dtype).to(device)

# Initialize the DiT, DiT-S with 12 layers, hidden size 384
transformer = SanaDiTS().to(dtype).to(device)

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

dataloader_train, dataloader_eval = load_IN1k256px(batch_size=bs)

optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)

# Step logger, calculates avg. sample processing and dataloading time
steplog = StepLogger()

# Eval: images
def eval_images(seed = 42, guidance_scales = [2, 7]):
    images = []
    for prompt in tqdm(eval_prompts, "eval_images"):
        img = pil_concat([
            generate(prompt, guidance_scale=guidance_scale, seed=seed, latent_dim=latent_dim)
            for guidance_scale in guidance_scales
        ])
        img = pil_add_text(img, prompt, position=(0,0))
        images.append(img)

    return transforms.ToPILImage()(
        make_grid(
            [transforms.ToTensor()(img) for img in images],
            pad_value=1,
            nrow=3
        )
    )

# Eval: clip score
def eval_clipscore():
    images = [
        generate(
            prompt,
            # 4x4 latents -> 128x128px images
            latent_dim=latent_dim,
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
step_dl_time, step_dl_start = None, None
for e in range(epochs):
    for labels, latents in dataloader_train:
        step += 1
        steplog.step_start()
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
            step_time, dl_time = steplog.get_avg_step_time(num_steps=20), steplog.get_avg_dl_time(num_steps=20)

            print(f"step {step} epoch {epoch:.2f} loss: {loss.item()} step_time: {step_time:.2f} dl_time: {dl_time:.2f} ")
            wandb.log({"step": step, "step_time": step_time, "dl_time": dl_time, "epoch": epoch, "loss_train": loss.item()})
            step_dl_time = 0

        if step % 500 == 0:
            wandb.log({"step": step, "epoch": epoch, "eval_images": wandb.Image(eval_images())})

        steplog.step_end()

    # eval after each epoch
    el = eval_loss()
    print(f"eval step {step} epoch {epoch:.2f} loss: {el:.2f}")
    wandb.log({"step": step, "epoch": epoch, "loss_eval": el, "eval_clipscore": eval_clipscore()})

    # save after each epoch
    transformer.save_pretrained(f"IN1k-256px_e{e}")

# log a big 10x10 gallery 
wandb.log({"final_gallery": wandb.Image(eval_images())})

wandb.finish()



