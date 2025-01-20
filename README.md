# Hana
A [Sana](https://nvlabs.github.io/Sana/)-like diffusion model

<div align="center" border-radius="10px">
  <img src="assets/hana_logo_small.png" width="60%"/>
</div>

## WIP 

| Model&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Text Encoder | AE | Transformer | Dataset&nbsp;&nbsp;&nbsp;&nbsp; | Model | Code | Loss | Samples |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Alpha-1 | [Gemma2 2b](https://huggingface.co/google/gemma-2-2b)  | [dc-ae-f32c32-sana-1.0](https://huggingface.co/mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers) | SanaTransformer2DModel (158.18M) 7 layers instead of 28 | [MNIST](https://huggingface.co/datasets/g-ronimo/MNIST-latents_dc-ae-f32c32-sana-1.0) <br/> 5 epochs <br/> LR 1e-4 <br/>300k steps <br/>BS 1 | [Model](https://huggingface.co/g-ronimo/hana-small_MNIST-5e) | [Code](https://github.com/geronimi73/Hana/tree/main/Alpha-1) | [0.958](https://wandb.ai/g-ronimo/Hana/runs/zf38z5gx?nw=nwusergronimo) | ![images_eval_15565_b120cf6385fa11612684](https://github.com/user-attachments/assets/96cc0930-47f7-4dfd-aeec-f8f361e75466)





