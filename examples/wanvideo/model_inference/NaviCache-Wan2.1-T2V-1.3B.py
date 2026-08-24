"""
NaviCache Inference Script for Wan2.1-T2V-1.3B.

Training-free, offline-calibration-free inference acceleration via NaviCache
(ICML 2026, "Test-Time Self-Calibration Caching for Video Generation").

NaviCache caches the entire DiT output residual (``output - input``) per
denoising step and, on low-drift steps, skips the whole transformer forward,
reconstructing ``input + cached_residual``. A scalar Kalman filter calibrated
online tracks the ratio ``Δoutput / Δinput`` and gates the skip decision, so the
method adapts to the current sample without any fixed residual threshold.

Usage:
    python NaviCache-Wan2.1-T2V-1.3B.py

Controls:
    - thresh:      Accumulated predicted-error threshold (lower = fewer skips,
                   higher fidelity). Wan2.1 fast/mid/slow ≈ 0.07/0.05/0.04.
    - align_steps: Initial exact-compute steps (Kalman calibration warm-up).
    - process_noise / measurement_noise: Kalman Q and R.
    - num_inference_steps, cfg_scale, seed: standard Wan2.1 sampling controls.
"""

import torch
from tqdm import tqdm
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import set_to_torch_norm
from diffsynth.models.wan_video_dit import FLASH_ATTN_3_AVAILABLE, FLASH_ATTN_2_AVAILABLE, SAGE_ATTN_AVAILABLE
from diffsynth.models.navicache import NaviCache


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

# Model
model_id = "Wan-AI/Wan2.1-T2V-1.3B"

# Generation
prompt = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

num_inference_steps = 15  # official Wan2.1 default
cfg_scale = 5.0
seed = 0
num_frames = 81
height = 480
width = 832

# NaviCache (Wan2.1 recommended values: fast/mid/slow ≈ 0.07/0.05/0.04)
thresh = 0.05                # accumulated predicted-error threshold (skip/miss boundary)
align_steps = 10             # initial exact-compute steps for Kalman calibration
process_noise = 0.05         # Kalman process-noise covariance Q
measurement_noise = 0.05     # Kalman measurement-noise covariance R

# Output
output_path = "navicache_output.mp4"


# ═══════════════════════════════════════════════════════════════
# Load model
# ═══════════════════════════════════════════════════════════════

print("Loading model...")
pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
    ],
    tokenizer_config=ModelConfig(model_id=model_id, origin_file_pattern="google/umt5-xxl/"),
)

dit = pipe.dit
vae = pipe.vae
scheduler = pipe.scheduler
device = pipe.device
dtype = pipe.torch_dtype

print(f"  DiT blocks: {len(dit.blocks)}")
print(f"  DiT dim: {dit.dim}")
print(f"  Patch size: {dit.patch_size}")

# Use PyTorch's fused RMSNorm kernel to avoid float32 temporaries.
set_to_torch_norm([dit])
print("  RMSNorm: torch_norm enabled")
print(f"  Attention: FA3={FLASH_ATTN_3_AVAILABLE}, FA2={FLASH_ATTN_2_AVAILABLE}, Sage={SAGE_ATTN_AVAILABLE}")


# ═══════════════════════════════════════════════════════════════
# Encode prompts
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_prompt(pipe, prompt: str) -> torch.Tensor:
    """Encode a text prompt with the pipeline's T5 tokenizer + text encoder."""
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb

print("Encoding prompts...")
ctx_posi = encode_prompt(pipe, prompt)       # [1, seq_len, text_dim]
ctx_nega = encode_prompt(pipe, negative_prompt)


# ═══════════════════════════════════════════════════════════════
# Initialize latents
# ═══════════════════════════════════════════════════════════════

print("Initializing latents...")
z_dim = vae.model.z_dim
latent_frames = (num_frames - 1) // 4 + 1     # temporal compression
latent_h = height // vae.upsampling_factor     # spatial compression
latent_w = width // vae.upsampling_factor

shape = (1, z_dim, latent_frames, latent_h, latent_w)
latents = pipe.generate_noise(shape, seed=seed, rand_device="cpu")
latents = latents.to(dtype=dtype, device=device)

print(f"  Latent shape: {latents.shape}")


# ═══════════════════════════════════════════════════════════════
# Setup scheduler + NaviCache
# ═══════════════════════════════════════════════════════════════

scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
print(f"  Timesteps: {len(scheduler.timesteps)} steps, "
      f"range [{scheduler.timesteps[0]:.1f}, {scheduler.timesteps[-1]:.1f}]")

# Wrap the DiT. cfg=True pairs the conditional and unconditional CFG forwards so
# the skip/compute decision is made on the positive branch only (mirroring how
# the RAS scripts restrict selection to the positive branch).
use_cfg = (cfg_scale != 1.0)
navicache = NaviCache(
    dit,
    thresh=thresh,
    align_steps=align_steps,
    num_inference_steps=num_inference_steps,
    cfg=use_cfg,
    process_noise=process_noise,
    measurement_noise=measurement_noise,
)


# ═══════════════════════════════════════════════════════════════
# Free GPU memory: offload models not needed during denoising
# ═══════════════════════════════════════════════════════════════

# After encoding, T5 text encoder is no longer needed on GPU.
# VAE is only needed for final decoding. Move both to CPU.
for model_attr, name in [
    (pipe.text_encoder, "text_encoder"),
    (pipe.vae, "vae"),
    (getattr(pipe, "image_encoder", None), "image_encoder"),
    (getattr(pipe, "motion_controller", None), "motion_controller"),
]:
    if model_attr is not None:
        model_attr.to("cpu")
        print(f"  Moved {name} to CPU")

torch.cuda.empty_cache()
torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════
# Denoising loop with NaviCache
# ═══════════════════════════════════════════════════════════════

dit.eval()

print(f"\nDenoising ({num_inference_steps} steps, NaviCache thresh={thresh}, "
      f"align={align_steps})...")

n_computed = 0
n_skipped = 0

# torch.inference_mode() disables autograd tracking (avoids OOM on the DiT stack).
with torch.inference_mode():
    for progress_id, timestep in enumerate(tqdm(scheduler.timesteps)):
        t = timestep.unsqueeze(0).to(dtype=dtype, device=device)

        # --- Positive (conditional) forward ---
        noise_posi = navicache.forward(x=latents, timestep=t, context=ctx_posi)
        if navicache.should_compute:
            n_computed += 1
        else:
            n_skipped += 1

        # --- Classifier-free guidance ---
        if use_cfg:
            noise_nega = navicache.forward(x=latents, timestep=t, context=ctx_nega)
            noise_pred = noise_nega + cfg_scale * (noise_posi - noise_nega)
        else:
            noise_pred = noise_posi

        # --- Scheduler step ---
        latents = scheduler.step(
            noise_pred,
            scheduler.timesteps[progress_id],
            latents,
        )

total = n_computed + n_skipped
print(f"\nNaviCache: {n_computed}/{total} steps computed exactly, "
      f"{n_skipped}/{total} skipped ({100 * n_skipped / max(total, 1):.0f}%)")


# ═══════════════════════════════════════════════════════════════
# Decode
# ═══════════════════════════════════════════════════════════════

# The DiT is no longer needed; free it before the (VRAM-hungry) VAE decode.
dit.to("cpu")
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("Decoding video...")
vae.to(device)
video = vae.batched_tiled_decode(
    latents, device=device,
    tile_size=(30, 52), tile_stride=(15, 26),
    tile_batch_size=2,
)
video = pipe.vae_output_to_video(video)
save_video(video, output_path, fps=15, quality=5)
print(f"Video saved to: {output_path}")
