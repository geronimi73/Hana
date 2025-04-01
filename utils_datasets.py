import torch, os, random
from datasets import load_dataset
from torch.utils.data import RandomSampler, DistributedSampler, DataLoader
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ShapeBatchingDataset(torch.utils.data.Dataset):
    def __init__(
        self, hf_dataset, 
        text_enc, tokenizer, 
        bs, ddp=False, device=None, dtype=None, 
        col_id="image_id", col_label="label", col_latent="latent", col_latentshape="latent_shape",
        seed=42
    ):
        self.hf_dataset = hf_dataset
        self.col_label, self.col_latent, self.col_id, self.col_latentshape = col_label, col_latent, col_id, col_latentshape
        self.bs = bs
        self.text_enc, self.tokenizer =  text_enc, tokenizer
        self.tokenizer.padding_side = "right"
        self.prompt_len = 50

        if device is None and dtype is None:
            self.device, self.dtype = text_enc.device, text_enc.dtype
        else:
            self.device, self.dtype = device, dtype

        if ddp: self.sampler = DistributedSampler(hf_dataset, shuffle=True, seed=seed)
        else: self.sampler = RandomSampler(hf_dataset, generator=torch.manual_seed(seed))

        def collate_fn(x): return x

        # preload samples with DataLoader, because accessing the hf dataset is expensive (90% of time spent in formatting.py:144(extract_row))
        self.dataloader = DataLoader(
            hf_dataset, sampler=self.sampler, collate_fn=collate_fn, batch_size=bs*2, 
            num_workers=2, 
            prefetch_factor=10
        )
    
    def encode_prompts(self, prompts):
        prompts_tok = self.tokenizer(
            prompts, padding="max_length", truncation=True, max_length=self.prompt_len, return_attention_mask=True, return_tensors="pt"
        )
        with torch.no_grad():
            prompts_encoded = self.text_enc(**prompts_tok.to(self.text_enc.device))
        return prompts_encoded.last_hidden_state, prompts_tok.attention_mask

    def __iter__(self):
        samples_by_shape, epoch = {}, 0

        while True:
            if isinstance(self.sampler, DistributedSampler): self.sampler.set_epoch(epoch)

            for samples in self.dataloader:
                for sample in samples:
                    shape = tuple(sample[self.col_latentshape])
        
                    # group items by shape
                    if not shape in samples_by_shape: samples_by_shape[shape] = []
                    samples_by_shape[shape].append(sample)
        
                    # once we have enough items of a given shape -> collate and yield a batch
                    if len(samples_by_shape[shape]) == self.bs: 
                        yield self.prepare_batch(samples_by_shape[shape], shape)
                        samples_by_shape[shape] = []
            epoch += 1
                
    def prepare_batch(self, items, shape):
        latent_shape = [len(items)]+list(shape)
        labels = [
            # random pick between md2, qwen2 and smolvlm
            item[self.col_label][random.randint(1, len(item[self.col_label])-1)]
            for item in items
        ]
        latents = torch.Tensor([item[self.col_latent] for item in items]).reshape(latent_shape)
        latents = latents.to(self.dtype).to(self.device)

        label_embs, label_atnmasks = self.encode_prompts(labels)

        return labels, latents, label_embs, label_atnmasks

    def __len__(self): return len(self.sampler) // self.bs


