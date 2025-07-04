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
    generate, 
    encode_prompt, 
    add_random_noise,
    pil_clipscore,
    SanaDiTBSmolLM360M,
    StepLogger,
    pil_concat,
    pil_add_text,
    latent_to_PIL,
    load_IN1k256px,
    load_IN1k256px_AR,
    load_CC12MIN21K256px
)

lr = 5e-4
bs = 256
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
    "a dog with blue eyes",
    "a white cat with purple eyes",
    "a black car in a beautiful field of flowers",
]
prompt_maxlen = 100
add_random_noise = partial(add_random_noise, dist="normal")  # use lognormal sigmas, default is uniform

te_repo = "HuggingFaceTB/SmolLM2-360M"
sana_repo = "Efficient-Large-Model/Sana_600M_1024px_diffusers"

set_seed(42)

# Choose device and dtype
dtype = torch.bfloat16
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

# Load the text encoder and AE
tokenizer = AutoTokenizer.from_pretrained(te_repo, torch_dtype=dtype)
tokenizer.pad_token = tokenizer.eos_token
text_encoder = AutoModel.from_pretrained(te_repo, torch_dtype=dtype).to(device)
dcae = AutoencoderDC.from_pretrained(sana_repo, subfolder="vae", torch_dtype=dtype).to(device)

# Initialize the DiT, DiT-S with 12 layers, hidden size 384
transformer = SanaDiTBSmolLM360M(atn_dropout=0.0).to(dtype).to(device)

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

# dataloader_train, dataloader_eval = load_IN1k256px(batch_size=bs, label_dropout=0.0)
dataloader_train, dataloader_eval = load_CC12MIN21K256px(batch_size=bs, label_dropout=0.1)
# dataloader_train, dataloader_eval = load_IN1k256px_AR(
#     tokenizer=tokenizer,
#     text_encoder=text_encoder,
#     batch_size=bs,
#     label_dropout=0.1,
# )

print(f"#batches train: {len(dataloader_train)}")
print(f"#samples train: ~{len(dataloader_train) * bs}")
print(f"#batches test: {len(dataloader_eval)}")
print(f"#samples test: ~{len(dataloader_eval) * bs}")

optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr)

# Step logger, calculates avg. sample processing and dataloading time
steplog = StepLogger()

# Eval: images
def eval_images(guidance_scales = [2, 7]):
    images = []
    for prompt in tqdm(eval_prompts, "eval_images"):
        img = pil_concat([
            generate(
                prompt, 
                guidance_scale=guidance_scale, 
                seed=seed, 
                latent_dim=latent_dim, 
                num_steps=20,
                max_prompt_tok = prompt_maxlen,
            )
            for seed in [42, 6945, 4009]
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
    transformer.eval()

    images = [
        generate(
            prompt,
            # 4x4 latents -> 128x128px images
            latent_dim=latent_dim,
            guidance_scale=7,
            num_steps=20,
            max_prompt_tok = prompt_maxlen,
        )
        for prompt in tqdm(eval_prompts, "eval_clipscore")
    ]

    transformer.train()

    return pil_clipscore(images, eval_prompts)

# Eval: loss
def eval_loss():
    transformer.eval()
    losses = []

    for batch_num, (labels, latents) in tqdm(enumerate(dataloader_eval), "eval_loss"):
    # for batch_num, (labels, latents, prompts_emb, prompts_atnmask) in tqdm(enumerate(dataloader_eval), "eval_loss"):
        # Encode prompts
        prompts_emb, prompts_atnmask = encode_prompt(labels, tokenizer, text_encoder, max_length=prompt_maxlen)

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

    transformer.train()

    return sum(losses)/len(losses)

# Setup logging 
wandb.init(
    project="IN128px", 
    name=f"{transformer_params:.1f}M_BS-{bs}_LR-{lr}_{epochs}-epochs"
).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".json"))

# TRAIN
transformer.train()
step = 0 
for e in range(epochs):
    # for labels, latents, prompts_emb, prompts_atnmask in dataloader_train:
    for labels, latents in dataloader_train:

        # debug: save zerobatch
        if step == 0:
            latents = latents.to(dtype).to(device)
            images = latent_to_PIL(latents[:min(32, len(labels))], dcae)
            [ 
                pil_add_text(img, labels[img_num], position=(0,0)).save(f"zerobatch_{img_num}.png")
                for img_num, img in enumerate(images)
            ]

        step += 1
        steplog.step_start()
        epoch = step/len(dataloader_train)

        # Encode prompts
        prompts_emb, prompts_atnmask = encode_prompt(labels, tokenizer, text_encoder, max_length=prompt_maxlen)

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
        
        optimizer.zero_grad()
        loss = F.mse_loss(noise_pred, noise - latents)
        loss.backward()    
        grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
        optimizer.step()
    
        if step % 10 == 0:
            step_time, dl_time = steplog.get_avg_step_time(num_steps=20), steplog.get_avg_dl_time(num_steps=20)

            print(f"step {step} epoch {epoch:.2f} loss: {loss.item()} step_time: {step_time:.2f} dl_time: {dl_time:.2f} ")
            wandb.log({"step": step, "step_time": step_time, "dl_time": dl_time, "epoch": epoch, "loss_train": loss.item(), "grad_norm": grad_norm})

        if step % 500 == 0:
            wandb.log({"step": step, "epoch": epoch, "eval_images": wandb.Image(eval_images())})

        steplog.step_end()

    # eval after each epoch
    el = eval_loss()
    print(f"eval step {step} epoch {epoch:.2f} loss: {el:.2f}")
    wandb.log({"step": step, "epoch": epoch, "loss_eval": el, "eval_clipscore": eval_clipscore()})

    # save after each epoch
    # transformer.save_pretrained(f"IN1k-256px_e{e}")
    max_retries, retry_delay = 100, 20
    for attempt in range(max_retries):
        try:
            # transformer.push_to_hub("g-ronimo/HanaDitB-0528-SmolLM2-360M-256px", variant=f"epoch{e}", private=True)
            transformer.push_to_hub("g-ronimo/HanaDitB_0530_beta-7", variant=f"epoch{e}", private=True)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Upload failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                print(e)
                time.sleep(retry_delay)
            else:
                print(f"Max retries reached. Skipping upload")


# log a big 10x10 gallery 
wandb.log({"final_gallery": wandb.Image(eval_images())})

wandb.finish()



