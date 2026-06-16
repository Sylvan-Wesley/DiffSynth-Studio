import glob
import os
from pathlib import Path

import torch
from PIL import Image
from modelscope import dataset_snapshot_download

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


MODEL_DIR = Path(
    os.environ.get(
        "WAN_MODEL_DIR",
        "/home/weixinyuan/AISys1/Wan2.1/Wan2.1-I2V-14B-480P",
    )
).expanduser()

INPUT_IMAGE = Path(os.environ.get("WAN_INPUT_IMAGE", "data/examples/wan/input_image.jpg"))
OUTPUT_VIDEO = os.environ.get("WAN_OUTPUT_VIDEO", "video_Wan2.1-I2V-14B-480P.mp4")


def require_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Required model file does not exist: {path}")
    return str(path)


def require_dir(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"Required model directory does not exist: {path}")
    return str(path)


def wan_dit_shards(model_dir: Path) -> list[str]:
    shards = sorted(glob.glob(str(model_dir / "diffusion_pytorch_model-*.safetensors")))
    if not shards:
        single_file = model_dir / "diffusion_pytorch_model.safetensors"
        if single_file.is_file():
            shards = [str(single_file)]
    if not shards:
        raise FileNotFoundError(
            "No DiT safetensors were found. Expected files like "
            f"{model_dir / 'diffusion_pytorch_model-00001-of-00007.safetensors'}"
        )
    return shards


print(f"Using Wan model directory: {MODEL_DIR}")

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    redirect_common_files=False,
    model_configs=[
        ModelConfig(path=wan_dit_shards(MODEL_DIR)),
        ModelConfig(path=require_file(MODEL_DIR / "models_t5_umt5-xxl-enc-bf16.pth")),
        ModelConfig(path=require_file(MODEL_DIR / "Wan2.1_VAE.pth")),
        ModelConfig(path=require_file(MODEL_DIR / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
    ],
    tokenizer_config=ModelConfig(path=require_dir(MODEL_DIR / "google" / "umt5-xxl")),
)

if not INPUT_IMAGE.is_file():
    dataset_snapshot_download(
        dataset_id="DiffSynth-Studio/examples_in_diffsynth",
        local_dir="./",
        allow_file_pattern="data/examples/wan/input_image.jpg",
    )

image = Image.open(INPUT_IMAGE).convert("RGB")

video = pipe(
    prompt="一艘小船正勇敢地乘风破浪前行。蔚蓝的大海波涛汹涌，白色的浪花拍打着船身，但小船毫不畏惧，坚定地驶向远方。阳光洒在水面上，闪烁着金色的光芒，为这壮丽的场景增添了一抹温暖。镜头拉近，可以看到船上的旗帜迎风飘扬，象征着不屈的精神与冒险的勇气。这段画面充满力量，激励人心，展现了面对挑战时的无畏与执着。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=image,
    seed=0,
    tiled=True,
)
save_video(video, OUTPUT_VIDEO, fps=15, quality=5)
print(f"Saved video to: {OUTPUT_VIDEO}")
