from datasets import load_dataset
from torch.utils.data import RandomSampler, DistributedSampler
from tqdm import tqdm
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ShapeBatchingDataset(torch.utils.data.Dataset):
    def __init__(
        self, hf_dataset, 
        # text_enc, tokenizer, 
        bs, ddp=False, col_id="image_id", col_label="caption", col_latent="ae_latent", col_latentshape="ae_latent_shape",
        seed=42
    ):
        self.hf_dataset = hf_dataset
        self.col_label, self.col_latent, self.col_id, self.col_latentshape = col_label, col_latent, col_id, col_latentshape
        self.bs = bs
        # TODO
        # self.text_enc, self.tokenizer =  text_enc, tokenizer
        # self.tokenizer.padding_side = "right"
        self.prompt_len = 50

        if ddp: self.sampler = DistributedSampler(hf_dataset, shuffle=True, seed=seed)
        else: self.sampler = RandomSampler(hf_dataset, generator=torch.manual_seed(seed))

    def encode_prompts(self, prompts): return prompts

    def __iter__(self):
        samples_by_shape = {}
        
        for sample_idx in self.sampler:
            sample = self.hf_dataset[sample_idx]
            shape = tuple(sample[self.col_latentshape])

            # group items by shape
            if not shape in samples_by_shape: samples_by_shape[shape] = []
            samples_by_shape[shape].append(sample)

            # once we have enough items of a given shape -> collate and yield a batch
            if len(samples_by_shape[shape]) == self.bs: 
                yield self.prepare_batch(samples_by_shape[shape], shape)
                samples_by_shape[shape] = []
                
    def prepare_batch(self, items, shape):
        latent_shape = [len(items)]+list(shape)
        labels = [item[self.col_label] for item in items]
        latents = torch.Tensor([item[self.col_latent] for item in items]).reshape(latent_shape)

        return labels, latents
    
    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)