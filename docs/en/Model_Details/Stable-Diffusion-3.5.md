# Stable Diffusion 3.5

Stable Diffusion 3.5 is a Stability AI text-to-image model family based on a Multimodal Diffusion Transformer (MMDiT). Unlike Stable Diffusion v1.5 and SDXL, SD 3.5 is not a UNet + DDIM pipeline; DiffSynth-Studio exposes it through a dedicated `StableDiffusion35Pipeline`.

This initial integration is inference-only and delegates execution to Hugging Face Diffusers' `StableDiffusion3Pipeline`. Training, native DiffSynth checkpoint converters, ControlNet, IP-Adapter, and LoRA training are not enabled in this first milestone.

## Installation

Before performing model inference, install DiffSynth-Studio first.

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

For SD 3.5, you also need access to the gated Hugging Face model repository. Accept the model license on Hugging Face and run:

```shell
hf auth login
```

## Quick Start

```python
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
    cfg_scale=3.5,
    num_inference_steps=28,
    seed=0,
)
image.save("sd35_large.jpg")
```

## Low VRAM Inference

SD 3.5 uses three text encoders and a large transformer, so model CPU offload is recommended on most consumer GPUs.

```python
import torch
from diffsynth.pipelines.stable_diffusion_35 import StableDiffusion35Pipeline

pipe = StableDiffusion35Pipeline.from_pretrained(
    model_id="stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
    device="cuda",
    download_source="huggingface",
    enable_model_cpu_offload=True,
)

image = pipe(
    prompt="A detailed editorial photo of a small robot arranging flowers on a wooden desk",
    cfg_scale=3.5,
    num_inference_steps=28,
    seed=0,
)
image.save("sd35_large_low_vram.jpg")
```

For lower memory usage with reduced prompt fidelity, disable the T5 encoder:

```python
pipe = StableDiffusion35Pipeline.from_pretrained(
    model_id="stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
    device="cuda",
    download_source="huggingface",
    enable_model_cpu_offload=True,
    use_t5_encoder=False,
)
```

## Model Overview

| Model ID | Inference | Low VRAM Inference | Full Training | LoRA Training |
| - | - | - | - | - |
| [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_35/model_inference/stable-diffusion-3.5-large.py) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_35/model_inference_low_vram/stable-diffusion-3.5-large.py) | - | - |
| [stabilityai/stable-diffusion-3.5-large-turbo](https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo) | [code](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/stable_diffusion_35/model_inference/stable-diffusion-3.5-large-turbo.py) | - | - | - |

## Model Inference

The model is loaded via `StableDiffusion35Pipeline.from_pretrained`.

Input parameters include:

* `prompt`: Text prompt.
* `negative_prompt`: Negative prompt, defaults to an empty string.
* `prompt_2`, `prompt_3`: Optional prompts for the second and third text encoders.
* `negative_prompt_2`, `negative_prompt_3`: Optional negative prompts for the second and third text encoders.
* `cfg_scale`: Classifier-free guidance scale passed to Diffusers as `guidance_scale`. Use `3.5` for Large and `0.0` for Large Turbo.
* `height`, `width`: Output image size. Values are rounded up to multiples of 16.
* `seed`: Random seed.
* `rand_device`: Device used to create the random generator, defaults to `"cpu"`.
* `num_inference_steps`: Number of inference steps. Use `28` for Large and `4` for Large Turbo.
* `t5_sequence_length`: Max sequence length for the T5 text encoder, default `512`.
* `progress_bar_cmd`: Set to `None` to disable the Diffusers progress bar.

`from_pretrained` accepts:

* `model_id`: Hugging Face or ModelScope model ID.
* `model_config`: Optional DiffSynth `ModelConfig` for local path or explicit download control.
* `download_source`: `huggingface` or `modelscope`.
* `enable_model_cpu_offload`: Enable Diffusers model CPU offload.
* `enable_sequential_cpu_offload`: Enable lower-memory, slower sequential offload.
* `use_t5_encoder`: Disable the third text encoder when set to `False`.

## Training

Training is not implemented for `StableDiffusion35Pipeline` yet. Use this integration for inference only.
