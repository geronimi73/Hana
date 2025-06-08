import os
import torch
import torchvision.transforms as T
import time
import platform
import random
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.functional.multimodal import clip_score
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from operator import itemgetter
from datasets import load_dataset

# stop complaining
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def make_grid(images, rows=1, cols=None):
    if cols is None: cols = len(images)
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def load_imagenet_1k_vl_enriched_recaped():
    import requests, gzip, json
    from io import BytesIO
    
    # URL of the gzipped JSON file
    url = "https://huggingface.co/datasets/g-ronimo/imagenet-1k-vl-enriched-recaped/resolve/main/captions.json.gz"
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Check if the request was successful
    
    with gzip.GzipFile(fileobj=BytesIO(response.content)) as gz:
        data = json.loads(gz.read().decode('utf-8'))
    return data

def load_IN1k256px_AR(batch_size=512, batch_size_eval=256, label_dropout=0.1):
    splits_train = ["train_AR_1_to_1", "train_AR_3_to_4", "train_AR_4_to_3"]
    splits_eval = ["validation_AR_1_to_1", "validation_AR_3_to_4", "validation_AR_4_to_3"]

    ds = load_dataset("g-ronimo/IN1k256-AR-buckets-bfl16latents_dc-ae-f32c32-sana-1.0")

    dataloader_train = ImageNetARDataset(
        ds, 
        splits=splits_train, 
        bs=batch_size, 
        label_dropout=label_dropout,
        ddp=False,
    )

    dataloader_eval = ImageNetARDataset(
        ds, 
        splits=splits_eval, 
        bs=batch_size, 
        label_dropout=None,
        ddp=False
    )

    return dataloader_train, dataloader_eval

class ImageNetARDataset(torch.utils.data.Dataset):
    def __init__(
        self, hf_dataset, splits, bs, label_dropout=None, ddp=False, col_id="image_id", col_label="label", col_latent="latent"
    ):
        self.hf_dataset = hf_dataset
        self.bs = bs
        # each split is one aspect ratio
        self.splits = splits  
        self.col_label, self.col_latent, self.col_id = col_label, col_latent, col_id
        self.label_dropout = label_dropout

        # load md2, qwen2 and smolvlm captions
        self.in1k_recaps = load_imagenet_1k_vl_enriched_recaped()

        seed = 42

        # Create a dataloader for each split (=aspect ratio)
        self.dataloaders = {}
        self.samplers = {}
        for split in splits:
            if ddp: 
                self.samplers[split] = DistributedSampler(hf_dataset[split], shuffle=True, seed=seed)
            else: 
                self.samplers[split] = RandomSampler(hf_dataset[split], generator=torch.manual_seed(seed))
            self.dataloaders[split] = DataLoader(
                hf_dataset[split], sampler=self.samplers[split], collate_fn=self.collate, batch_size=bs, num_workers=4, prefetch_factor=2
            )

    def collate(self, items):
        labels = [
            # random pick between md2, qwen2 and smolvlm
            self.in1k_recaps[i[self.col_id]][random.randint(0, 2)]
            for i in items
        ]

        # drop 10% of the labels
        if self.label_dropout:
            labels = [ label if random.random() > self.label_dropout else "" for label in labels ]

        # latents shape [B, 1, 32, W, H] -> squeeze [B, 32, W, H]
        latents = torch.Tensor([i[self.col_latent] for i in items]).squeeze()

        return labels, latents
  
    def __iter__(self):
        # Reset iterators at the beginning of each epoch
        iterators = { split: iter(dataloader) for split, dataloader in self.dataloaders.items() }
        active_dataloaders = set(self.splits)  # Track exhausted dataloaders
        current_split_index = -1
        
        while active_dataloaders:
            # Round robin: change split on every iteration (=after every batch OR after we unsucc. tried to get a batch) 
            current_split_index = (current_split_index + 1) % len(self.splits)
            split = self.splits[current_split_index]

            # Skip if this dataloader is exhausted
            if split not in active_dataloaders: continue
            
            # Try to get the next batch
            try:
                labels, latents = next(iterators[split]) 

                yield labels, latents
            # dataloader is exhausted
            except StopIteration: active_dataloaders.remove(split)

    def set_epoch(self, epoch):
        for split in self.splits: self.samplers[split].set_epoch(epoch)

    def __len__(self):
        return sum([len(self.samplers[split]) for split in self.splits]) // self.bs

def SanaDiTBSmolLM360M(atn_dropout=0.1):
    from diffusers import SanaTransformer2DModel
    sana_repo = "Efficient-Large-Model/Sana_600M_1024px_diffusers"

    config = SanaTransformer2DModel.load_config(sana_repo, subfolder="transformer")
    config["num_layers"] = 12
    config["caption_channels"] = 960
    config["dropout"] = atn_dropout

    config["num_attention_heads"] = 12
    config["attention_head_dim"] = 64

    config["cross_attention_dim"] = 768
    config["num_cross_attention_heads"] = 12
    config["cross_attention_head_dim"] = 64

    # Sanity check
    hidden_dim = config["num_attention_heads"] * config["attention_head_dim"] 
    xatn_dim, xatn_heads, xatn_headdim  = itemgetter(
        "cross_attention_dim", 
        "num_cross_attention_heads",
        "cross_attention_head_dim",
    )(config)

    print("Model width:", hidden_dim)
    assert hidden_dim == xatn_dim, f"Mismatch, cross_attention_dim={xatn_dim}"
    assert hidden_dim == xatn_heads * xatn_headdim, f"Mismatch, {xatn_heads} * {xatn_headdim} = {xatn_heads * xatn_headdim}"

    return SanaTransformer2DModel.from_config(config)

def generate(prompt, transformer, tokenizer, text_encoder, dcae, num_steps = 10, latent_dim = [1, 32, 8, 8], guidance_scale = None, neg_prompt = "", seed=None, max_prompt_tok=50):
    device, dtype = transformer.device, transformer.dtype
    do_cfg = guidance_scale is not None

    # Encode the prompt, +neg. prompt if classifier free guidance (CFG)
    prompt_encoded, prompt_atnmask = encode_prompt(
        [prompt, neg_prompt] if do_cfg else prompt, 
        tokenizer, 
        text_encoder,
        max_length = max_prompt_tok
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
    
def encode_prompt(prompt, tokenizer, text_encoder, max_length=50, add_special_tokens=False, **kwargs):
    # lower case prompt! took a long time to find that this is necessary: https://github.com/huggingface/diffusers/blob/e8aacda762e311505ba05ae340af23b149e37af3/src/diffusers/pipelines/sana/pipeline_sana.py#L433
    tokenizer.padding_side = "right"
    if isinstance(prompt, list):
        prompt = [p.lower().strip() for p in prompt]
    elif isinstance(prompt, str):
        prompt = prompt.lower().strip()
    else:
        raise Exception(f"Unknown prompt type {type(prompt)}")         
    prompt_tok = tokenizer(prompt, return_tensors="pt", return_attention_mask=True, padding="max_length", truncation=True, max_length=max_length, add_special_tokens=add_special_tokens).to(text_encoder.device)
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

def pil_concat(images, horizontal=True):
    w, h = images[0].size
    combined = Image.new('RGB', (w * len(images), h) if horizontal else (w, h  * len(images)))
    for i, img in enumerate(images):
        combined.paste(img, (w * i, 0) if horizontal else (0, h * i))
    return combined

def pil_add_text(image, text, position=None, font_size=None, font_color=(255, 255, 255), 
                       font_path=None, stroke_width=1, stroke_fill=(0, 0, 0)):
    if font_path is None: 
        if platform.system() == "Darwin":
            font_path = "Times.ttc"
        else:
            font_path = "DejaVuSans.ttf"
    w, h = image.size
    if position is None: position = (w//10, h//10)
    if font_size is None: font_size = round(h*0.1)
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    font = ImageFont.truetype(font_path, font_size)

    draw.text(
        position,
        text,
        font=font,
        fill=font_color,
        stroke_width=stroke_width,
        stroke_fill=stroke_fill
    )
    
    return img_copy

class StepLogger:
    def __init__(self):
        self.step_times = []
        self.dl_times = []
        self.start_time, self.end_time = None, None

    def step_start(self):
        self.start_time = time.time()
        if self.end_time:
            self.dl_times.append(self.start_time - self.end_time)
            
    def step_end(self):
        self.end_time = time.time()
        self.step_times.append(self.end_time - self.start_time)        

    def get_avg_step_time(self, num_steps=None):
        if num_steps is None: num_steps = len(self.step_times)
        if not self.step_times or num_steps <= 0: return 0.0
        num_steps = min(num_steps, len(self.step_times))

        return sum(self.step_times[-num_steps:]) / num_steps

    def get_avg_dl_time(self, num_steps=None):
        if num_steps is None: num_steps = len(self.dl_times)
        if not self.dl_times or num_steps <= 0: return 0.0
        num_steps = min(num_steps, len(self.dl_times))

        return sum(self.dl_times[-num_steps:]) / num_steps

FMNIST_LABEL_TO_DESC = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag",
    9: "Ankle boot",
}
