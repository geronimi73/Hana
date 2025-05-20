import torch
import torchvision.transforms as T
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.functional.multimodal import clip_score

FMNIST_LABEL_TO_DESC = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag",
    9: "Ankle boot",
}

def generate(prompt, transformer, tokenizer, text_encoder, dcae, num_steps = 10, latent_dim = [1, 32, 8, 8], guidance_scale = None, neg_prompt = "", seed=None):
    device, dtype = transformer.device, transformer.dtype
    do_cfg = guidance_scale is not None

    # Encode the prompt, +neg. prompt if classifier free guidance (CFG)
    prompt_encoded, prompt_atnmask = encode_prompt(
        [prompt, neg_prompt] if do_cfg else prompt, 
        tokenizer, 
        text_encoder
    )
        
    # Divide 1000 -> 0 in equally sized steps
    timesteps = torch.linspace(1000, 0, num_steps + 1, device=device, dtype=dtype)
    
    # Noise level. 1.0 -> 0.0 in equally sized steps
    sigmas = timesteps / 1000
    
    latent = torch.randn(
        latent_dim, 
        generator=torch.manual_seed(seed) if seed else None
    ).to(dtype).to(device)
    
    for t, sigma_prev, sigma_next, steps_left in zip(
        timesteps, 
        sigmas[:-1], 
        sigmas[1:], 
        range(num_steps, 0, -1)
    ):
        t = t[None].to(device)

        # DiT predicts noise
        with torch.no_grad():
            noise_pred = transformer(
                hidden_states = torch.cat([latent] * 2) if do_cfg else latent,
                timestep = torch.cat([t] * 2) if do_cfg else t,
                encoder_hidden_states=prompt_encoded,
                encoder_attention_mask=prompt_atnmask,
                return_dict=False
            )[0]

        if do_cfg:
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # Remove noise from latent
        latent = latent + (sigma_next - sigma_prev) * noise_pred 

    return latent_to_PIL(latent / dcae.config["scaling_factor"], dcae)

def add_random_noise(latents, timesteps=1000, dist="uniform"):
    assert dist in ["normal", "uniform"], f"Requested sigma dist. {dist} not supported"

    # batch size
    bs = latents.size(0)

    # gaussian noise
    noise = torch.randn_like(latents)

    # normal distributed sigmas
    if dist == "normal":
        sigmas = torch.randn((bs,)).sigmoid().to(latents.device)
    else:
        sigmas = torch.rand((bs,)).to(latents.device)
    # sigmas = torch.randn((bs,)).sigmoid().to(latents.device)
    
    timesteps = (sigmas * timesteps).to(latents.device)   # yes, `timesteps = sigmas * 1000`, let's keep it simple
    sigmas = sigmas.view([latents.size(0), *([1] * len(latents.shape[1:]))])
    
    latents_noisy = (1 - sigmas) * latents + sigmas * noise # (1-noise_level) * latent + noise_level * noise

    return latents_noisy.to(latents.dtype), timesteps, noise
    
def encode_prompt(prompt, tokenizer, text_encoder, max_length=50):
    # lower case prompt! took a long time to find that this is necessary: https://github.com/huggingface/diffusers/blob/e8aacda762e311505ba05ae340af23b149e37af3/src/diffusers/pipelines/sana/pipeline_sana.py#L433
    tokenizer.padding_side = "right"
    if isinstance(prompt, list):
        prompt = [p.lower().strip() for p in prompt]
    elif isinstance(prompt, str):
        prompt = prompt.lower().strip()
    else:
        raise Exception(f"Unknown prompt type {type(prompt)}")         
    prompt_tok = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, padding="max_length", truncation=True, max_length=max_length, add_special_tokens=True).to(text_encoder.device)
    with torch.no_grad():
        prompt_encoded=text_encoder(**prompt_tok)
    return prompt_encoded.last_hidden_state, prompt_tok.attention_mask

def latent_to_PIL(latent, ae):
    with torch.no_grad():
        image_out = ae.decode(latent).sample.to("cpu")
    
    if image_out.size(0) == 1:
        # Single image processing
        image_out = torch.clamp_(image_out[0,:], -1, 1)
        image_out = image_out * 0.5 + 0.5
        return T.ToPILImage()(image_out.float())
    else:
        images = []
        for img in image_out:
            img = torch.clamp_(img, -1, 1)
            img = img * 0.5 + 0.5
            images.append(T.ToPILImage()(img.float()))
        return images

def PIL_to_latent(images, ae):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        lambda x: x.to(dtype=ae.dtype)
    ])

    if not isinstance(images, (list, tuple)): images = [images]
    
    images_tensors = torch.cat([transform(image)[None] for image in images])
    
    with torch.no_grad():
        latent = ae.encode(images_tensors.to(ae.device))
    return latent.latent

def pil_clipscore(images, prompts, clip_model="openai/clip-vit-base-patch16"):
    images_tens = [pil_to_tensor(i) for i in images]
    with torch.no_grad():
        scores = clip_score(images_tens, prompts, model_name_or_path=clip_model).detach()
    return scores.item()
