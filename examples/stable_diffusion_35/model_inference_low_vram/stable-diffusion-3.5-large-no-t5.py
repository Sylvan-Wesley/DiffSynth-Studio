import torch
from diffsynth.pipelines.stable_diffusion_35 import StableDiffusion35Pipeline


pipe = StableDiffusion35Pipeline.from_pretrained(
    model_id="stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
    device="cuda",
    download_source="huggingface",
    enable_model_cpu_offload=True,
    use_t5_encoder=False,
)

image = pipe(
    prompt="A detailed editorial photo of a small robot arranging flowers on a wooden desk",
    seed=0,
    cfg_scale=3.5,
    num_inference_steps=28,
    height=1024,
    width=1024,
)
image.save("sd35_large_no_t5.jpg")
