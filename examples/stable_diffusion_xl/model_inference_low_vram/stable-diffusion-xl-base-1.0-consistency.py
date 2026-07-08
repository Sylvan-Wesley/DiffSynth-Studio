import argparse
import os
import torch
from diffsynth.core import ModelConfig
from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLConsistencyPipeline

DEFAULT_DMD_LORA_DIR = "~/DiffSynth-Studio/models/train/stable-diffusion-xl-base-1.0_dmd_laion_prompts"


def parse_args():
    parser = argparse.ArgumentParser(description="Run SDXL DMD consistency inference.")
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=1000,
        help="Training step used to build step-<N>-generator.safetensors.",
    )
    parser.add_argument(
        "--dmd_lora_dir",
        type=str,
        default=os.environ.get("DMD_LORA_DIR", DEFAULT_DMD_LORA_DIR),
        help="Directory containing DMD step-<N>-generator.safetensors checkpoints.",
    )
    parser.add_argument(
        "--dmd_generator_lora_path",
        type=str,
        default=os.environ.get("DMD_GENERATOR_LORA_PATH"),
        help="Optional explicit path to a *-generator.safetensors checkpoint.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="image_consistency.jpg",
        help="Where to save the generated image.",
    )
    return parser.parse_args()


def resolve_dmd_generator_lora_path(args):
    if args.dmd_generator_lora_path is not None:
        return os.path.expanduser(args.dmd_generator_lora_path)
    dmd_lora_dir = os.path.expanduser(args.dmd_lora_dir)
    return os.path.join(dmd_lora_dir, f"step-{args.checkpoint_step}-generator.safetensors")


args = parse_args()
DMD_GENERATOR_LORA_PATH = resolve_dmd_generator_lora_path(args)

vram_config = {
    "offload_dtype": torch.float32,
    "offload_device": "cpu",
    "onload_dtype": torch.float32,
    "onload_device": "cpu",
    "preparing_dtype": torch.float32,
    "preparing_device": "cuda",
    "computation_dtype": torch.float32,
    "computation_device": "cuda",
}
pipe = StableDiffusionXLConsistencyPipeline.from_pretrained(
    torch_dtype=torch.float32,
    model_configs=[
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder/model.safetensors", **vram_config),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="text_encoder_2/model.safetensors", **vram_config),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="unet/diffusion_pytorch_model.safetensors", **vram_config),
        ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="vae/diffusion_pytorch_model.safetensors", **vram_config),
    ],
    tokenizer_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer/"),
    tokenizer_2_config=ModelConfig(model_id="stabilityai/stable-diffusion-xl-base-1.0", origin_file_pattern="tokenizer_2/"),
    vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
)
if not os.path.exists(DMD_GENERATOR_LORA_PATH):
    raise FileNotFoundError(
        f"DMD generator LoRA not found: {DMD_GENERATOR_LORA_PATH}. "
        "Set DMD_GENERATOR_LORA_PATH to your *-generator.safetensors checkpoint."
    )
pipe.load_lora(pipe.unet, DMD_GENERATOR_LORA_PATH)

image = pipe(
    prompt="a photo of an astronaut riding a horse on mars",
    negative_prompt="",
    step_num=4,
    cfg_scale=1.0,
    height=1024,
    width=1024,
    seed=42,
    num_inference_steps=1000,
)
image.save(args.output_path)
