# Hana
A [Sana](https://nvlabs.github.io/Sana/)-like diffusion model

<div align="center" border-radius="10px">
  <img src="assets/hana_logo_small.png" width="60%"/>
</div>

## WIP 

| Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Text Encoder | AE | Transformer | Dataset&nbsp;&nbsp;&nbsp;&nbsp; | Compute | Model | Code | Loss | Samples |
| ----------- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------|--- | ------------------------------------------------------------------------ | ------------------------------------------------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Alpha-3     | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)   | [dc-ae-f32c32-sana-1.0](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers)         | SanaTransformer2DModel (158.18M) `num_layers=7`, `cross_attention_dim=1152`            | [MNIST](https://huggingface.co/datasets/g-ronimo/MNIST-latents_dc-ae-f32c32-sana-1.0) <br/>85660 steps,=150 epochs,<br/>BS 128                          | 1x4090, 8 hrs  | [Model](https://huggingface.co/g-ronimo/hana-small_MNIST-BATCHED)        | [Code](https://github.com/geronimi73/Hana/tree/main/Alpha-3)       | [0.833](https://wandb.ai/g-ronimo/Hana/runs/nahswxbq)        | ![media_images_images_eval_670_f1a015427c67c3e5933e](https://github.com/user-attachments/assets/3bd1dade-69c4-4b52-8725-66fbd3e9dce6)
| Alpha-2     | [Gemma2 2b](https://huggingface.co/google/gemma-2-2b)   | [dc-ae-f32c32-sana-1.0](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers)         | SanaTransformer2DModel (158.18M) 7 layers instead of 28            | [MNIST](https://huggingface.co/datasets/g-ronimo/MNIST-latents_dc-ae-f32c32-sana-1.0) <br/> 7940 steps,<br/>BS 128                          | 1x4090, 40'  | [Model](https://huggingface.co/g-ronimo/hana-small_MNIST-BATCHED)        | [Code](https://github.com/geronimi73/Hana/tree/main/Alpha-2)       | [0.933](https://wandb.ai/g-ronimo/Hana/runs/wbxx3k1y)        | ![images_eval_406_016a491efda29b0ea833](https://github.com/user-attachments/assets/8b367260-3a78-47a0-b6ce-65c95aee78fe)
| Alpha-1     | [Gemma2 2b](https://huggingface.co/google/gemma-2-2b)   | [dc-ae-f32c32-sana-1.0](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers)         | SanaTransformer2DModel (158.18M) 7 layers instead of 28            | [MNIST](https://huggingface.co/datasets/g-ronimo/MNIST-latents_dc-ae-f32c32-sana-1.0) <br/> 5 epochs,<br/>LR 1e-4,<br/>300k steps,<br/>BS 1 | 1x4090, 4 hours   | [Model](https://huggingface.co/g-ronimo/hana-small_MNIST-5e)             | [Code](https://github.com/geronimi73/Hana/tree/main/Alpha-1)       | [0.958](https://wandb.ai/g-ronimo/Hana/runs/zf38z5gx)        | ![images_eval_15565_b120cf6385fa11612684](https://github.com/user-attachments/assets/96cc0930-47f7-4dfd-aeec-f8f361e75466)









