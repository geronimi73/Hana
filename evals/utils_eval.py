import requests, random, torch

from pathlib import Path
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.functional.multimodal import clip_score
from tqdm import tqdm

from utils import generate, pil_add_text, make_grid

def get_random_prompts(n=100, seed=42, save=True, 
                      url="https://raw.githubusercontent.com/google-research/parti/main/PartiPrompts.tsv",
                      fn="random_parti_prompts.txt"
    ):
    "Download PartiPrompts.tsv and extract n random prompts"
    random.seed(seed)
    
    # Download and process in one step
    r = requests.get(url)
    lines = r.text.strip().split('\n')
    
    # Extract prompts (skip header)
    prompts = [line.split('\t')[0] for line in lines[1:]]
    
    # Select random prompts
    selected = random.sample(prompts, n)
    
    # Optionally save to file
    if save:
        with open(fn, 'w') as f:
            for p in selected: f.write(f"{p}\n")
    
    return selected

def load_prompts(fn="random_parti_prompts.txt"):
    "Load prompts from file as a list of strings"
    with open(fn, 'r') as f:
        return [line.strip() for line in f]

def pil_clipscore(images, prompts, clip_model="openai/clip-vit-base-patch16"):
    images_tens = [pil_to_tensor(i) for i in images]
    with torch.no_grad():
        scores = clip_score(images_tens, prompts, model_name_or_path=clip_model).detach()
    return scores.item()

def generate_images_seed(prompt, seed, cfgs, pipeline_components, inference_config):
    # list of unlabeled images
    imgs = [generate(prompt, guidance_scale=cfg, latent_seed=seed, **pipeline_components, **inference_config) for cfg in cfgs]
    # grid of images labeled with CFG and seed
    imgs_labeled = [pil_add_text(imgs[i], str(cfgs[i]), position=(0,imgs[0].height*8.5/10)) for i in range(len(imgs))]
    gallery = make_grid(imgs_labeled)
    gallery = pil_add_text(gallery, str(seed), position=(imgs[0].width*0.5,gallery.height*8.5/10))
    # grid of (unlabeled images)
    # imgs = make_grid(imgs)

    return imgs, gallery

def generate_images_prompts(prompts, seeds, cfgs, pipeline_components, inference_config):
    imgs = []
    gallery_prompts = []

    for prompt in tqdm(prompts):
        gallery_seeds = []
        
        for seed in seeds:
            # imgs_ = list of images, one for each cfg
            # imgs_labeled_ = single image, grid of images, one for each cfg
            imgs_, gallery = generate_images_seed(prompt, seed, cfgs, pipeline_components, inference_config)
            imgs.extend(imgs_)
            gallery_seeds.extend([gallery])
        # turn list of galleries into single img and add prompt
        gallery_seeds = make_grid(gallery_seeds, rows=1)
        gallery_seeds = pil_add_text(gallery_seeds, prompt, position=(0,0))
    
        gallery_prompts.extend([gallery_seeds])

    return imgs, gallery_prompts

