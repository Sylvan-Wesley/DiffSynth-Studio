"""
RAS (Region-Adaptive Sampling) Inference Script for Wan2.1-T2V-1.3B.

This script demonstrates training-free inference acceleration using RAS from:
  Liu et al., "Region-Adaptive Sampling for Diffusion Transformers", CVPR 2026.

RAS dynamically selects which spatial regions of the video latent to process
at each denoising step. Active regions go through the full DiT stack; inactive
regions receive a cheap "dumb" head-only update. KV-caching avoids recomputing
context projections for inactive tokens. This achieves ~2x speedup with minimal
quality loss.

Usage:
    python RAS-Wan2.1-T2V-1.3B.py

Controls:
    - ratio:     Fraction of tokens processed per step (0.25 default, lower = faster)
    - cfg_scale: Classifier-free guidance scale (5.0 default)
    - num_inference_steps: Total denoising steps
    - seed:      Random seed for reproducibility
    - enable_viz: Store selection masks for visualization
"""

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import selection_mask_to_grid, set_to_torch_norm
from diffsynth.models.wan_video_dit import FLASH_ATTN_3_AVAILABLE, FLASH_ATTN_2_AVAILABLE, SAGE_ATTN_AVAILABLE


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

# Model
model_id = "Wan-AI/Wan2.1-T2V-1.3B"

# Generation
prompt = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"

negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

num_inference_steps = 30
cfg_scale = 5.0
seed = 0
num_frames = 81
height = 480
width = 832

# RAS
ratio = 0.25                # fraction of tokens updated per step (1.0 = full, 0.25 = 4x fewer)
num_dense_steps = 3         # initial steps with full updates to warm KV caches
enable_viz = True           # store per-step selection masks for visualization

# Output
output_path = "ras_output.mp4"
viz_output_dir = "ras_masks"  # directory for mask visualization frames


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
# This replaces the custom x.float() path with a memory-efficient fused CUDA kernel.
set_to_torch_norm([dit])
print("  RMSNorm: torch_norm enabled")
print(f"  Attention: FA3={FLASH_ATTN_3_AVAILABLE}, FA2={FLASH_ATTN_2_AVAILABLE}, Sage={SAGE_ATTN_AVAILABLE}")


# ═══════════════════════════════════════════════════════════════
# Encode prompts
# ═══════════════════════════════════════════════════════════════

def encode_prompt(pipe, prompt: str) -> torch.Tensor:
    """Encode a text prompt using the pipeline's T5 tokenizer + text encoder."""
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

print(f"  Latent shape: {latents.shape}  (B={latents.shape[0]}, C={latents.shape[1]}, "
      f"F={latents.shape[2]}, H={latents.shape[3]}, W={latents.shape[4]})")


# ═══════════════════════════════════════════════════════════════
# Setup scheduler
# ═══════════════════════════════════════════════════════════════

scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
print(f"  Timesteps: {len(scheduler.timesteps)} steps, "
      f"range [{scheduler.timesteps[0]:.1f}, {scheduler.timesteps[-1]:.1f}]")


# ═══════════════════════════════════════════════════════════════
# Initialize RAS state
# ═══════════════════════════════════════════════════════════════

num_layers = len(dit.blocks)
# Patchify reduces spatial dims by patch_size[1:] (e.g. (2,2) → h/2, w/2 patches).
# S must be the number of tokens AFTER patchify, not the VAE latent dims.
S = latent_frames * (latent_h // dit.patch_size[1]) * (latent_w // dit.patch_size[2])
B = latents.shape[0]

# Per-layer KV caches for self-attention and cross-attention
# Separate caches for positive and negative CFG branches
kv_cache_posi = [{} for _ in range(num_layers)]
ctx_kv_cache_posi = [{} for _ in range(num_layers)]
kv_cache_nega = [{} for _ in range(num_layers)]
ctx_kv_cache_nega = [{} for _ in range(num_layers)]

# Starvation prevention: skip_list tracks cumulative skip penalty per token
skip_list = torch.zeros(B, S, device=device)
skip_k = torch.zeros(B, S, device=device)    # drop counter per token (0 = never skipped)

# Reset previous noise (first step falls back to L2 norm heuristic)
dit._prev_noise_tokens = None

# Pre-compute all-patches index for dense warm-up steps
all_patches = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

print(f"  Total tokens: {S}  (={latent_frames}×{latent_h//dit.patch_size[1]}×{latent_w//dit.patch_size[2]} patches)")
print(f"  Active tokens (ratio={ratio}): {int(S * ratio)}")
print(f"  Dense warm-up steps: {num_dense_steps}")
print(f"  Expected speedup: ~{1/ratio:.1f}x (after warm-up)")


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

# Report available GPU memory
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    used_gb = (total - free) / (1024**3)
    print(f"  GPU memory in use before denoising: {used_gb:.1f} GiB / {total/(1024**3):.1f} GiB")


# ═══════════════════════════════════════════════════════════════
# Denoising loop with RAS
# ═══════════════════════════════════════════════════════════════

dit.eval()
dit.clear_selection_masks()

print(f"\nDenoising ({num_inference_steps} steps, RAS ratio={ratio}, "
      f"dense warm-up={num_dense_steps})...")

# torch.inference_mode() disables autograd tracking entirely.
# Without it, PyTorch saves intermediate tensors for all 30 DiT blocks
# (~3 GB/block) to support backward(), causing OOM on 79 GB GPU.
# inference_mode() is preferred over no_grad() because it also disables
# version-counter bumps and view tracking, giving a small additional
# memory saving.
with torch.inference_mode():
    for progress_id, timestep in enumerate(tqdm(scheduler.timesteps)):
        t = timestep.unsqueeze(0).to(dtype=dtype, device=device)

        # Dense steps: process ALL tokens to warm KV caches.
        # Sparse steps: let model auto-select which tokens to process.
        is_dense = progress_id < num_dense_steps
        selected_patches = all_patches if is_dense else None

        # --- Positive (conditional) forward ---
        noise_posi = dit.forward(
            x=latents,
            timestep=t,
            context=ctx_posi,
            kv_cache=kv_cache_posi,
            ctx_kv_cache=ctx_kv_cache_posi,
            skip_list=skip_list,
            skip_k=skip_k,
            selected_patches=selected_patches,
            ratio=ratio,
            enable_debug_masks=enable_viz,
        )

        # Free transient memory from posi forward before running nega forward.
        # The posi KV cache must persist for the next step, but intermediate
        # activations from the posi forward can be released.
        torch.cuda.empty_cache()

        # --- Classifier-free guidance ---
        if cfg_scale != 1.0:
            noise_nega = dit.forward(
                x=latents,
                timestep=t,
                context=ctx_nega,
                kv_cache=kv_cache_nega,
                ctx_kv_cache=ctx_kv_cache_nega,
                skip_list=skip_list,
                skip_k=skip_k,
                selected_patches=selected_patches,
                ratio=ratio,
                enable_debug_masks=False,   # only record masks for positive branch
            )
            noise_pred = noise_nega + cfg_scale * (noise_posi - noise_nega)
        else:
            noise_pred = noise_posi

        # --- Scheduler step ---
        latents = scheduler.step(
            noise_pred,
            scheduler.timesteps[progress_id],
            latents,
        )


# ═══════════════════════════════════════════════════════════════
# Decode
# ═══════════════════════════════════════════════════════════════

print("Decoding video...")
vae.to(device)
video = vae.decode(
    latents, device=device,
    tiled=True, tile_size=(30, 52), tile_stride=(15, 26),
)
video = pipe.vae_output_to_video(video)
save_video(video, output_path, fps=15, quality=5)
print(f"Video saved to: {output_path}")


# ═══════════════════════════════════════════════════════════════
# Visualize selection masks
# ═══════════════════════════════════════════════════════════════

if enable_viz:
    print("\nVisualizing selection masks...")
    masks = dit.get_selection_masks()
    print(f"  Recorded {len(masks)} per-step masks")

    # Each mask entry: (timestep_value, mask_tensor [B, S], grid_size (f, h, w))
    for idx, (t_val, mask, (f, h, w)) in enumerate(masks):
        # Convert [B, S] bool mask → [B, 1, f, h, w] float spatial grid
        grid = selection_mask_to_grid(mask, (f, h, w))

        # Stats: what fraction of tokens were selected this step?
        frac_selected = mask.float().mean().item()
        print(f"  Step {idx:2d} (t={t_val:6.1f}): "
              f"{frac_selected*100:4.1f}% tokens selected  "
              f"grid shape={list(grid.shape)}  "
              f"grid_size=({f},{h},{w})")

    # Save per-frame mask grids for the first batch element
    # Upsample spatial dimensions (h, w) to match output resolution for overlay
    import os
    os.makedirs(viz_output_dir, exist_ok=True)
    for idx, (t_val, mask, (f, h, w)) in enumerate(masks):
        grid = selection_mask_to_grid(mask, (f, h, w))  # [1, 1, f, h, w]
        # Average over frames for a summary frame mask
        frame_mask = grid[0, 0].mean(dim=0).cpu().numpy()  # [h, w] in patch space
        # Save as numpy for external visualization
        np.save(f"{viz_output_dir}/mask_step_{idx:02d}_t_{t_val:.0f}.npy", frame_mask)

    print(f"  Mask arrays saved to: {viz_output_dir}/")
    print(f"  To visualize as heatmap overlay, upsample each {h}×{w} mask to {height}×{width}")
