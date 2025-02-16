from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import pil_to_tensor
from torchmetrics.functional.multimodal import clip_score
from tqdm import tqdm
from scipy import integrate
import torch
import torchvision.transforms as T
import json
import requests
import gc
import platform
import matplotlib.pyplot as plt

# Custom dataset, got fed up with HF Datasets performance
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, col_label="label", col_latent="latent"):
        self.labels = []
        self.latents = []

        for item in tqdm(hf_dataset, "Converting HF to torch Dataset"):
            self.labels.append(item[col_label])
            self.latents.append(torch.Tensor(item[col_latent]))
        self.features = [col_label, col_latent]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return dict(label=self.labels[idx], latent=self.latents[idx])


# condensed from https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py
def get_sigma_schedule(steps, timesteps_train = 1000, flow_shift = 1.0):
    sigmas = torch.linspace(1, 0, steps + 1)
    sigmas = flow_shift * sigmas / (1 + (flow_shift - 1) * sigmas)
    timesteps = sigmas * timesteps_train
    return sigmas, timesteps

def get_rnd_sigmas(num_samples, dist="normal"):
    if dist == "normal":
        sigmas = torch.randn((num_samples,)).sigmoid()
    elif dist == "uniform":
        sigmas = torch.rand((num_samples,))
    elif dist in ["beta", "beta-high"]:
        if dist == "beta": alpha, beta = 1, 2.5
        else: alpha, beta = 2.5, 1        
        beta_dist = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(beta))
        sigmas = beta_dist.sample([num_samples])
    else:
        raise Exception(f"unknown distribution {dist}")
    return sigmas

# source: https://github.com/crowsonkb/k-diffusion/blob/8018de0b43da8d66617f3ef10d3f2a41c1d78836/k_diffusion/sampling.py#L247
def linear_multistep_coeff(order, t, i, j):
    if order - 1 > i:
        raise ValueError(f'Order {order} too high for step {i}')
    def fn(tau):
        prod = 1.
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod
    return integrate.quad(fn, t[i], t[i + 1], epsrel=1e-4)[0]

def plot_density(x, title=""):
    plt.figure(figsize=(5, 3))
    plt.hist(x, bins=50, density=True, alpha=0.7)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def pil_add_text(image, text, position=None, font_size=None, font_color=(255, 255, 255), 
                       font_path=None, stroke_width=1, stroke_fill=(0, 0, 0)):
    if font_path is None: 
        if platform.system() == "Darwin":
            font_path = "Times.ttc"
        else:
            font_path = "DejaVuSans.ttf"
    w, h = image.size
    if position is None: position = (w//10, h//10)
    if font_size is None: font_size = round(h*0.2)
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

def pil_clipscore(images, prompts, clip_model="openai/clip-vit-base-patch16"):
    images_tens = [pil_to_tensor(i) for i in images]
    with torch.no_grad():
        scores = clip_score(images_tens, prompts, model_name_or_path=clip_model).detach()
    return scores.item()

# DC-AE scaling factor, see https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers/blob/main/vae/config.json
dcae_scalingf = 0.41407

# input: Tensor [B, C, W, H]
# output: PIL image(s), list, or single if B==1
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

# input: PIL image(s), list or single 
# output: Tensor [B, C, W, H]
def PIL_to_latent(images, ae):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        lambda x: x.to(dtype=torch.bfloat16)
    ])

    if not isinstance(images, (list, tuple)): images = [images]
    
    images_tensors = torch.cat([transform(image)[None] for image in images])
    
    with torch.no_grad():
        latent = ae.encode(images_tensors.to(ae.device))
    return latent.latent

def make_grid(images, rows=1, cols=None):
    if cols is None: cols = len(images)
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

# stolen from HF https://github.com/huggingface/diffusers/blob/5b1dcd15848f6748c6cec978ef962db391c4e4cd/src/diffusers/training_utils.py#L295
def free_memory():
    """
    Runs garbage collection. Then clears the cache of the available accelerator.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif is_torch_npu_available():
        torch_npu.npu.empty_cache()
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()

def encode_prompt(prompt, tokenizer, text_encoder):
    # lower case prompt! took a long time to find that this is necessary: https://github.com/huggingface/diffusers/blob/e8aacda762e311505ba05ae340af23b149e37af3/src/diffusers/pipelines/sana/pipeline_sana.py#L433
    tokenizer.padding_side = "right"
    prompt = prompt.lower().strip()
    prompt_tok = tokenizer(prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=300, add_special_tokens=True).to(text_encoder.device)
    with torch.no_grad():
        prompt_encoded=text_encoder(**prompt_tok)
    return prompt_encoded.last_hidden_state, prompt_tok.attention_mask

def load_imagenet_labels():
    raw_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    response = requests.get(raw_url)
    imagenet_labels = json.loads(response.text)
    return imagenet_labels

mnist_labels = {i: str(i) for i in range(10)}

cifar10_labels = {
    0: "airplane",
    1: "automobile",
    2: "bird", 
    3: "cat",
    4: "deer", 
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

fmnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

tinyin_labels = {0: 'goldfish, Carassius auratus',
    1: 'European fire salamander, Salamandra salamandra',
    2: 'bullfrog, Rana catesbeiana',
    3: 'tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui',
    4: 'American alligator, Alligator mississipiensis',
    5: 'boa constrictor, Constrictor constrictor',
    6: 'trilobite',
    7: 'scorpion',
    8: 'black widow, Latrodectus mactans',
    9: 'tarantula',
    10: 'centipede',
    11: 'koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus',
    12: 'jellyfish',
    13: 'brain coral',
    14: 'snail',
    15: 'sea slug, nudibranch',
    16: 'American lobster, Northern lobster, Maine lobster, Homarus americanus',
    17: 'spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish',
    18: 'black stork, Ciconia nigra',
    19: 'king penguin, Aptenodytes patagonica',
    20: 'albatross, mollymawk',
    21: 'dugong, Dugong dugon',
    22: 'Yorkshire terrier',
    23: 'golden retriever',
    24: 'Labrador retriever',
    25: 'German shepherd, German shepherd dog, German police dog, alsatian',
    26: 'standard poodle',
    27: 'tabby, tabby cat',
    28: 'Persian cat',
    29: 'Egyptian cat',
    30: 'cougar, puma, catamount, mountain lion, painter, panther, Felis concolor',
    31: 'lion, king of beasts, Panthera leo',
    32: 'brown bear, bruin, Ursus arctos',
    33: 'ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle',
    34: 'grasshopper, hopper',
    35: 'walking stick, walkingstick, stick insect',
    36: 'cockroach, roach',
    37: 'mantis, mantid',
    38: "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk",
    39: 'monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
    40: 'sulphur butterfly, sulfur butterfly',
    41: 'sea cucumber, holothurian',
    42: 'guinea pig, Cavia cobaya',
    43: 'hog, pig, grunter, squealer, Sus scrofa',
    44: 'ox',
    45: 'bison',
    46: 'bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis',
    47: 'gazelle',
    48: 'Arabian camel, dromedary, Camelus dromedarius',
    49: 'orangutan, orang, orangutang, Pongo pygmaeus',
    50: 'chimpanzee, chimp, Pan troglodytes',
    51: 'baboon',
    52: 'African elephant, Loxodonta africana',
    53: 'lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens',
    54: 'abacus',
    55: "academic gown, academic robe, judge's robe",
    56: 'altar',
    57: 'backpack, back pack, knapsack, packsack, rucksack, haversack',
    58: 'bannister, banister, balustrade, balusters, handrail',
    59: 'barbershop',
    60: 'barn',
    61: 'barrel, cask',
    62: 'basketball',
    63: 'bathtub, bathing tub, bath, tub',
    64: 'beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon',
    65: 'beacon, lighthouse, beacon light, pharos',
    66: 'beaker',
    67: 'beer bottle',
    68: 'bikini, two-piece',
    69: 'binoculars, field glasses, opera glasses',
    70: 'birdhouse',
    71: 'bow tie, bow-tie, bowtie',
    72: 'brass, memorial tablet, plaque',
    73: 'bucket, pail',
    74: 'bullet train, bullet',
    75: 'butcher shop, meat market',
    76: 'candle, taper, wax light',
    77: 'cannon',
    78: 'cardigan',
    79: 'cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM',
    80: 'CD player',
    81: 'chest',
    82: 'Christmas stocking',
    83: 'cliff dwelling',
    84: 'computer keyboard, keypad',
    85: 'confectionery, confectionary, candy store',
    86: 'convertible',
    87: 'crane',
    88: 'dam, dike, dyke',
    89: 'desk',
    90: 'dining table, board',
    91: 'dumbbell',
    92: 'flagpole, flagstaff',
    93: 'fly',
    94: 'fountain',
    95: 'freight car',
    96: 'frying pan, frypan, skillet',
    97: 'fur coat',
    98: 'gasmask, respirator, gas helmet',
    99: 'go-kart',
    100: 'gondola',
    101: 'hourglass',
    102: 'iPod',
    103: 'jinrikisha, ricksha, rickshaw',
    104: 'kimono',
    105: 'lampshade, lamp shade',
    106: 'lawn mower, mower',
    107: 'lifeboat',
    108: 'limousine, limo',
    109: 'magnetic compass',
    110: 'maypole',
    111: 'military uniform',
    112: 'miniskirt, mini',
    113: 'moving van',
    114: 'neck brace',
    115: 'obelisk',
    116: 'oboe, hautboy, hautbois',
    117: 'organ, pipe organ',
    118: 'parking meter',
    119: 'pay-phone, pay-station',
    120: 'picket fence, paling',
    121: 'pill bottle',
    122: "plunger, plumber's helper",
    123: 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria',
    124: 'poncho',
    125: 'pop bottle, soda bottle',
    126: "potter's wheel",
    127: 'projectile, missile',
    128: 'punching bag, punch bag, punching ball, punchball',
    129: 'refrigerator, icebox',
    130: 'remote control, remote',
    131: 'rocking chair, rocker',
    132: 'rugby ball',
    133: 'sandal',
    134: 'school bus',
    135: 'scoreboard',
    136: 'sewing machine',
    137: 'snorkel',
    138: 'sock',
    139: 'sombrero',
    140: 'space heater',
    141: "spider web, spider's web",
    142: 'sports car, sport car',
    143: 'steel arch bridge',
    144: 'stopwatch, stop watch',
    145: 'sunglasses, dark glasses, shades',
    146: 'suspension bridge',
    147: 'swimming trunks, bathing trunks',
    148: 'syringe',
    149: 'teapot',
    150: 'teddy, teddy bear',
    151: 'thatch, thatched roof',
    152: 'torch',
    153: 'tractor',
    154: 'triumphal arch',
    155: 'trolleybus, trolley coach, trackless trolley',
    156: 'turnstile',
    157: 'umbrella',
    158: 'vestment',
    159: 'viaduct',
    160: 'volleyball',
    161: 'water jug',
    162: 'water tower',
    163: 'wok',
    164: 'wooden spoon',
    165: 'comic book',
    166: 'reel',
    167: 'guacamole',
    168: 'ice cream, icecream',
    169: 'ice lolly, lolly, lollipop, popsicle',
    170: 'goose',
    171: 'drumstick',
    172: 'plate',
    173: 'pretzel',
    174: 'mashed potato',
    175: 'cauliflower',
    176: 'bell pepper',
    177: 'lemon',
    178: 'banana',
    179: 'pomegranate',
    180: 'meat loaf, meatloaf',
    181: 'pizza, pizza pie',
    182: 'potpie',
    183: 'espresso',
    184: 'bee',
    185: 'apron',
    186: 'pole',
    187: 'Chihuahua',
    188: 'alp',
    189: 'cliff, drop, drop-off',
    190: 'coral reef',
    191: 'lakeside, lakeshore',
    192: 'seashore, coast, seacoast, sea-coast',
    193: 'acorn',
    194: 'broom',
    195: 'mushroom',
    196: 'nail',
    197: 'chain',
    198: 'slug',
    199: 'orange'
 }