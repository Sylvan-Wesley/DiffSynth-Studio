"""
Benchmark: RAS vs Original Wan2.1-T2V-1.3B Inference Time Comparison.

Runs the same denoising loop with identical inputs twice:
  1. Full inference  (kv_cache=None) — all tokens through all DiT blocks every step.
  2. RAS inference   (kv_cache + ratio) — only selected tokens through DiT blocks.

Both paths use WanModel.forward() directly for an apples-to-apples comparison.

Usage:
    python benchmark_RAS_vs_Wan.py

Controls:
    - ratio:              Fraction of tokens processed per RAS step
    - num_dense_steps:    Initial full-update steps to warm KV caches
    - num_inference_steps: Total denoising steps
    - cfg_scale:          Classifier-free guidance scale
    - seed:               Random seed
"""

import time
import torch
import numpy as np
from tqdm import tqdm
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import set_to_torch_norm, update_noise_cache


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

model_id = "Wan-AI/Wan2.1-T2V-1.3B"

prompt = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

num_inference_steps = 50  # official Wan2.1 default
cfg_scale = 5.0
seed = 0
num_frames = 81
height = 480
width = 832

# RAS settings
ratio = 0.25
num_dense_steps = 3
dumb_update = "Previous"  # "Previous" | "Zero"

# Output
output_full = "benchmark_full_output.mp4"
output_ras  = "benchmark_ras_output.mp4"


# ═══════════════════════════════════════════════════════════════
# Shared setup: model, prompts, scheduler
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
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

print(f"  DiT blocks: {len(dit.blocks)}, dim: {dit.dim}, patch_size: {dit.patch_size}")

# Use PyTorch's fused RMSNorm kernel to avoid float32 temporaries.
set_to_torch_norm([dit])
print("  RMSNorm: torch_norm enabled")


def encode_prompt(pipe, prompt: str) -> torch.Tensor:
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


print("Encoding prompts...")
ctx_posi = encode_prompt(pipe, prompt)
ctx_nega = encode_prompt(pipe, negative_prompt)

# Latent dimensions
z_dim = vae.model.z_dim
latent_frames = (num_frames - 1) // 4 + 1
latent_h = height // vae.upsampling_factor
latent_w = width // vae.upsampling_factor
shape = (1, z_dim, latent_frames, latent_h, latent_w)
# Patchify reduces spatial dims by patch_size[1:] — S is tokens AFTER patchify.
S = latent_frames * (latent_h // dit.patch_size[1]) * (latent_w // dit.patch_size[2])
B = 1
num_layers = len(dit.blocks)

# Scheduler
scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
timesteps = scheduler.timesteps
print(f"  Timesteps: {len(timesteps)} steps")

# Offload unused models to CPU to free GPU memory for KV caches
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

if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    used_gb = (total - free) / (1024**3)
    print(f"  GPU memory in use before denoising: {used_gb:.1f} GiB / {total/(1024**3):.1f} GiB")


# ═══════════════════════════════════════════════════════════════
# Helper: run a single denoising step
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_step(latents, t, kv_cache_posi, ctx_kv_cache_posi,
                 kv_cache_nega, ctx_kv_cache_nega,
                 skip_list, skip_k, selected_patches, ratio_val,
                 prev_posi_noise_tokens=None, prev_nega_noise_tokens=None,
                 prev_guided_noise_tokens=None):
    """Run one denoising step and return the noise plus the next per-condition RAS caches."""
    # Positive branch
    noise_posi, noise_posi_tokens = dit.forward(
        x=latents, timestep=t, context=ctx_posi,
        kv_cache=kv_cache_posi, ctx_kv_cache=ctx_kv_cache_posi,
        skip_list=skip_list, skip_k=skip_k,
        selected_patches=selected_patches,
        ratio=ratio_val, dumb_update=dumb_update,
        enable_debug_masks=False,
        use_heuristics="prev_noise",
        prev_noise_tokens=prev_guided_noise_tokens,
        dumb_noise_tokens=prev_posi_noise_tokens,
        return_noise_tokens=True,
    )

    # Free transient posi activations before nega forward
    torch.cuda.empty_cache()

    # CFG
    if cfg_scale != 1.0:
        # Reuse the positive branch's selection so both CFG branches process
        # the same tokens (and the skip record updates once per step, not twice).
        nega_selected = (
            dit.get_last_selected_patches()
            if selected_patches is None else selected_patches
        )
        noise_nega, noise_nega_tokens = dit.forward(
            x=latents, timestep=t, context=ctx_nega,
            kv_cache=kv_cache_nega, ctx_kv_cache=ctx_kv_cache_nega,
            skip_list=skip_list, skip_k=skip_k,
            selected_patches=nega_selected,
            ratio=ratio_val, dumb_update=dumb_update,
            enable_debug_masks=False,
            prev_noise_tokens=prev_guided_noise_tokens,
            dumb_noise_tokens=prev_nega_noise_tokens,
            return_noise_tokens=True,
        )
        noise_pred = noise_nega + cfg_scale * (noise_posi - noise_nega)
        guided_noise_tokens = noise_nega_tokens + cfg_scale * (
            noise_posi_tokens - noise_nega_tokens
        )
        active_patches = nega_selected
    else:
        noise_pred = noise_posi
        guided_noise_tokens = noise_posi_tokens
        active_patches = dit.get_last_selected_patches()

    if kv_cache_posi is not None:
        # Scatter only the active predictions into each retained cache so
        # inactive entries keep their prior per-condition values.
        prev_posi_noise_tokens = update_noise_cache(
            prev_posi_noise_tokens, active_patches, noise_posi_tokens,
        )
        if cfg_scale != 1.0:
            prev_nega_noise_tokens = update_noise_cache(
                prev_nega_noise_tokens, active_patches, noise_nega_tokens,
            )
        prev_guided_noise_tokens = update_noise_cache(
            prev_guided_noise_tokens, active_patches, guided_noise_tokens,
        )
    return (noise_pred, prev_posi_noise_tokens, prev_nega_noise_tokens,
            prev_guided_noise_tokens)


# ═══════════════════════════════════════════════════════════════
# Run 1: Full inference (kv_cache=None → all tokens every step)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("RUN 1: Full inference (all tokens, all steps)")
print("=" * 60)

# Fresh latents
latents_full = pipe.generate_noise(shape, seed=seed, rand_device="cpu")
latents_full = latents_full.to(dtype=dtype, device=device)
dit.eval()
torch.cuda.synchronize()
t_start = time.perf_counter()

for progress_id, timestep in enumerate(tqdm(timesteps, desc="Full")):
    t = timestep.unsqueeze(0).to(dtype=dtype, device=device)

    # kv_cache=None → full eval path in WanModel.forward()
    noise_pred, _, _, _ = denoise_step(
        latents=latents_full, t=t,
        kv_cache_posi=None, ctx_kv_cache_posi=None,
        kv_cache_nega=None, ctx_kv_cache_nega=None,
        skip_list=None, skip_k=None,
        selected_patches=None, ratio_val=1.0,
    )

    latents_full = scheduler.step(
        noise_pred, timesteps[progress_id], latents_full,
    )

torch.cuda.synchronize()
t_full = time.perf_counter() - t_start
print(f"  Full inference time: {t_full:.2f}s  ({t_full/len(timesteps)*1000:.0f} ms/step)")


# ═══════════════════════════════════════════════════════════════
# Run 2: RAS inference (kv_cache + ratio → sparse updates)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(f"RUN 2: RAS inference (ratio={ratio}, dense warm-up={num_dense_steps})")
print("=" * 60)

# Fresh latents (identical noise, same seed)
latents_ras = pipe.generate_noise(shape, seed=seed, rand_device="cpu")
latents_ras = latents_ras.to(dtype=dtype, device=device)
prev_posi_noise_tokens = None
prev_nega_noise_tokens = None
prev_guided_noise_tokens = None

# RAS state
kv_cache_posi  = [{} for _ in range(num_layers)]
ctx_kv_cache_posi  = [{} for _ in range(num_layers)]
kv_cache_nega  = [{} for _ in range(num_layers)]
ctx_kv_cache_nega  = [{} for _ in range(num_layers)]
skip_list = torch.zeros(B, S, device=device)
skip_k    = torch.zeros(B, S, device=device)
all_patches = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

dit.clear_selection_masks()

torch.cuda.synchronize()
t_start = time.perf_counter()

for progress_id, timestep in enumerate(tqdm(timesteps, desc="RAS ")):
    t = timestep.unsqueeze(0).to(dtype=dtype, device=device)

    # First num_dense_steps: full update to warm caches
    is_dense = progress_id < num_dense_steps
    sel = all_patches if is_dense else None

    (noise_pred, prev_posi_noise_tokens, prev_nega_noise_tokens,
     prev_guided_noise_tokens) = denoise_step(
        latents=latents_ras, t=t,
        kv_cache_posi=kv_cache_posi, ctx_kv_cache_posi=ctx_kv_cache_posi,
        kv_cache_nega=kv_cache_nega, ctx_kv_cache_nega=ctx_kv_cache_nega,
        skip_list=skip_list, skip_k=skip_k,
        selected_patches=sel, ratio_val=ratio,
        prev_posi_noise_tokens=prev_posi_noise_tokens,
        prev_nega_noise_tokens=prev_nega_noise_tokens,
        prev_guided_noise_tokens=prev_guided_noise_tokens,
    )

    latents_ras = scheduler.step(
        noise_pred, timesteps[progress_id], latents_ras,
    )

torch.cuda.synchronize()
t_ras = time.perf_counter() - t_start
print(f"  RAS inference time: {t_ras:.2f}s  ({t_ras/len(timesteps)*1000:.0f} ms/step)")


# ═══════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════

speedup = t_full / t_ras
time_saved = t_full - t_ras
dense_time_est = (num_dense_steps / num_inference_steps) * t_full
sparse_time_est = t_ras - dense_time_est * (t_ras / t_full)  # rough
effective_speedup_sparse = (t_full - dense_time_est) / max(sparse_time_est, 0.001)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  {'Full inference':<30s} {t_full:>8.2f}s  ({t_full/len(timesteps)*1000:>5.0f} ms/step)")
print(f"  {'RAS inference':<30s} {t_ras:>8.2f}s  ({t_ras/len(timesteps)*1000:>5.0f} ms/step)")
print(f"  {'Time saved':<30s} {time_saved:>8.2f}s")
print(f"  {'Overall speedup':<30s} {speedup:>7.2f}x")
print(f"  {'Sparse-step speedup (excl warm-up)':<30s} {effective_speedup_sparse:>7.2f}x")
print()

# Per-step breakdown estimate
print(f"  Steps 0-{num_dense_steps-1} (dense warm-up):  ~{dense_time_est:.1f}s total  (full speed)")
print(f"  Steps {num_dense_steps}-{num_inference_steps-1} (RAS sparse):  ~{t_ras - dense_time_est * (t_ras/t_full):.1f}s total  (~{1/ratio:.0f}x sparse speedup)")
print()


# ═══════════════════════════════════════════════════════════════
# Decode and save both videos (optional — comment out to skip)
# ═══════════════════════════════════════════════════════════════

print("Decoding videos...")
vae.to(device)

video_full = vae.decode(
    latents_full, device=device,
    tiled=True, tile_size=(30, 52), tile_stride=(15, 26),
)
video_full = pipe.vae_output_to_video(video_full)
save_video(video_full, output_full, fps=15, quality=5)
print(f"  Full output: {output_full}")

video_ras = vae.decode(
    latents_ras, device=device,
    tiled=True, tile_size=(30, 52), tile_stride=(15, 26),
)
video_ras = pipe.vae_output_to_video(video_ras)
save_video(video_ras, output_ras, fps=15, quality=5)
print(f"  RAS output:  {output_ras}")

print("\nDone. Compare the two videos to assess quality difference.")
