import torch
from diffsynth.pipelines.stable_diffusion_35 import StableDiffusion35Pipeline


pipe = StableDiffusion35Pipeline.from_pretrained(
    model_id="stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
    device="cuda",
    download_source="huggingface",
)

image = pipe(
    prompt="A capybara holding a sign that reads Hello World",
    seed=0,
    cfg_scale=3.5,
    num_inference_steps=28,
    height=1024,
    width=1024,
)
image.save("sd35_large.jpg")
