import torch
import torchvision.transforms as T
from PIL import Image

# DC-AE scaling factor, see https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers/blob/main/vae/config.json
dcae_scalingf = 0.41407

def latent_to_PIL(latent, ae):
    with torch.no_grad():
        image_out = ae.decode(latent).sample.squeeze().to("cpu")    
    image_out = torch.clamp_(image_out, -1, 1)    # clamp, because output of is AE sometimes out of [-1;1] for some reason
    image_out = image_out * 0.5 + 0.5 # normalize to 0-1    
    return T.ToPILImage()(image_out.float())  

def PIL_to_latent(image, ae):
    transform = T.Compose([
        T.Resize(1024, antialias=True),
        # This will center crop from the resized image
        T.CenterCrop(1024),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        lambda x: x.to(dtype=torch.bfloat16)
    ])

    image_tensor = transform(image)[None].to(device)

    with torch.no_grad():
        latent = ae.encode(image_tensor)
    return latent.latent

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def encode_prompt(prompt, tokenizer, text_encoder):
    # lower case prompt! took a long time to find that this is necessary: https://github.com/huggingface/diffusers/blob/e8aacda762e311505ba05ae340af23b149e37af3/src/diffusers/pipelines/sana/pipeline_sana.py#L433
    tokenizer.padding_side = "right"
    prompt = prompt.lower().strip()
    prompt_tok = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=300, add_special_tokens=True).to(text_encoder.device)
    with torch.no_grad():
        prompt_encoded=text_encoder(**prompt_tok)
    return prompt_encoded.last_hidden_state, prompt_tok.attention_mask

# prompt_encoded, prompt_atnmask = encode_prompt("a red balloon", tokenizer, text_encoder)
# prompt_encoded.shape, prompt_atnmask.shape