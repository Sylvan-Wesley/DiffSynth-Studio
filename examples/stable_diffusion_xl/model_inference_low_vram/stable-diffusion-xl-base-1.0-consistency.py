import argparse
import os
import torch
from diffsynth.core import ModelConfig
from diffsynth.pipelines.stable_diffusion_xl import StableDiffusionXLConsistencyPipeline

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

DEFAULT_MODEL_BASE_DIR = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "~/DiffSynth-Studio/models")
DEFAULT_SDXL_MODEL_DIR = os.path.join(DEFAULT_MODEL_BASE_DIR, "stabilityai/stable-diffusion-xl-base-1.0")
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
        "--sdxl_model_dir",
        type=str,
        default=os.environ.get("SDXL_MODEL_DIR", DEFAULT_SDXL_MODEL_DIR),
        help="Local directory containing SDXL text encoders, UNet, VAE, and tokenizers.",
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
        "--inference_dtype",
        type=str,
        default=os.environ.get("SDXL_INFERENCE_DTYPE", "bfloat16"),
        choices=("float32", "bfloat16", "float16"),
        help="Precision used for SDXL consistency inference.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="image_consistency.jpg",
        help="Where to save the generated image.",
    )
    return parser.parse_args()


def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))


def require_path(path):
    path = expand_path(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required local model path does not exist: {path}")
    return path


def resolve_dmd_generator_lora_path(args):
    if args.dmd_generator_lora_path is not None:
        return expand_path(args.dmd_generator_lora_path)
    dmd_lora_dir = expand_path(args.dmd_lora_dir)
    return os.path.join(dmd_lora_dir, f"step-{args.checkpoint_step}-generator.safetensors")


def parse_dtype(dtype_name):
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]


args = parse_args()
SDXL_MODEL_DIR = expand_path(args.sdxl_model_dir)
DMD_GENERATOR_LORA_PATH = resolve_dmd_generator_lora_path(args)
INFERENCE_DTYPE = parse_dtype(args.inference_dtype)

vram_config = {
    "offload_dtype": INFERENCE_DTYPE,
    "offload_device": "cpu",
    "onload_dtype": INFERENCE_DTYPE,
    "onload_device": "cpu",
    "preparing_dtype": INFERENCE_DTYPE,
    "preparing_device": "cuda",
    "computation_dtype": INFERENCE_DTYPE,
    "computation_device": "cuda",
}
pipe = StableDiffusionXLConsistencyPipeline.from_pretrained(
    torch_dtype=INFERENCE_DTYPE,
    model_configs=[
        ModelConfig(path=require_path(os.path.join(SDXL_MODEL_DIR, "text_encoder/model.safetensors")), **vram_config),
        ModelConfig(path=require_path(os.path.join(SDXL_MODEL_DIR, "text_encoder_2/model.safetensors")), **vram_config),
        ModelConfig(path=require_path(os.path.join(SDXL_MODEL_DIR, "unet/diffusion_pytorch_model.safetensors")), **vram_config),
        ModelConfig(path=require_path(os.path.join(SDXL_MODEL_DIR, "vae/diffusion_pytorch_model.safetensors")), **vram_config),
    ],
    tokenizer_config=ModelConfig(path=require_path(os.path.join(SDXL_MODEL_DIR, "tokenizer"))),
    tokenizer_2_config=ModelConfig(path=require_path(os.path.join(SDXL_MODEL_DIR, "tokenizer_2"))),
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
