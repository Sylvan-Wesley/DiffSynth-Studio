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
from diffsynth.models.wan_video_dit import selection_mask_to_grid, set_to_torch_norm, update_noise_cache
from diffsynth.models.wan_video_dit import FLASH_ATTN_3_AVAILABLE, FLASH_ATTN_2_AVAILABLE, SAGE_ATTN_AVAILABLE


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

# RAS
ratio = 0.25                # fraction of tokens updated per step (1.0 = full, 0.25 = 4x fewer)
num_dense_steps = 3         # initial steps with full updates to warm KV caches
enable_viz = True           # store per-step selection masks for visualization
dumb_update = "Previous"

# MotionCache (SkyReels-V2 motion-aware token selection; Xu et al. 2026). Toggles
# the heuristic passed to select_region. Dict keys: weight_norm_mode ("mean"
# default, SkyReels), weight_floor (max_rescale only), mc_phase1_steps
# (uniform-weight warm-up, SkyReels token_phase1_update_count).
use_motion_cache = False            # set True to enable MotionCache selection
motion_cache_weight_norm_mode = "mean"   # "mean" | "max" | "max_rescale" (SkyReels default)
motion_cache_weight_floor = 0.6          # max_rescale floor (SkyReels value)
motion_cache_phase1_steps = 3            # uniform-weight warm-up steps before motion weights
use_heuristics = (
    {
        "name": "MotionCache",
        "weight_norm_mode": motion_cache_weight_norm_mode,
        "weight_floor": motion_cache_weight_floor,
        "mc_phase1_steps": motion_cache_phase1_steps,
    }
    if use_motion_cache else "prev_noise"
)

# VRAM (limited-GPU support; ported from vbench_eval_RAS_vs_Wan.py).
# The pipeline's default text-encoder seq_len is 512. T5 attention is O(L^2)
# and the cross-attention KV cache is [seq_len x dim] per layer, persisting for
# the whole denoising loop — capping seq_len cuts both. Prompts here are short,
# so 256 is safe.
text_seq_len = 256
# VAE spatial tiles per forward during decode (None = all tiles at once).
# Lower if VRAM is tight; the feathered-blend accumulation buffers stay on CPU.
tile_batch_size = 2

viz_mask_mode = "per_frame" # how selection masks are saved:
                            #   "frame_avg"  → one [h, w] map per step (averaged over frames) [default]
                            #   "per_frame"  → one [h, w] map per (step, frame)   mask_step_XX_f_KK_t_...
                            #   "full_grid"  → one [f, h, w] grid per step        mask_step_XX_f_all_t_...

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

# Cap the text-encoder sequence length (T5 attention is O(L^2), and the
# cross-attention KV cache persists across the whole denoising loop).
if text_seq_len != 512:
    print(f"  text_encoder seq_len: 512 -> {text_seq_len}")
pipe.tokenizer.seq_len = text_seq_len

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

@torch.no_grad()
def encode_prompt(pipe, prompt: str) -> torch.Tensor:
    """Encode a text prompt with the pipeline's T5 tokenizer + text encoder.

    no_grad() is essential here: the T5 encoder is 24 layers of full (non-flash)
    attention over a seq_len-padded sequence; without it, autograd retains every
    layer's activations and the encoder alone can consume tens of GB.
    """
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb

print("Encoding prompts...")
if torch.cuda.is_available():
    print(f"  GPU allocated after model load: "
          f"{torch.cuda.memory_allocated() / 1024**3:.1f} GiB")
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
skip_k = torch.zeros(B, S, device=device) - num_inference_steps    # drop counter per token (0 = never skipped)

# Per-condition token-noise caches, updated only at active regions each step.
# Each CFG branch's dumb fill carries forward its OWN condition's previous
# prediction (posi→posi, nega→nega), so `noise_pred = nega + cfg*(posi - nega)`
# is a genuine guided combination on every token. `prev_guided_noise_tokens`
# is that CFG combo and drives select_region, so selection follows the
# scheduler's actual trajectory.
prev_posi_noise_tokens = None
prev_nega_noise_tokens = None
prev_guided_noise_tokens = None
# Full (all-token) CFG-combined predictions, used as the MotionCache drift
# reference. Unlike prev_guided_noise_tokens (the scattered noise cache, which
# keeps stale values at inactive tokens), these carry a fresh value for every
# token every step, so the inter-step drift is non-zero everywhere and no frame
# is ever locked out of selection.
prev_guided_full = None
prev_guided_full_prev = None

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

# MotionCache per-token accumulator (Eq 12): zeros [B, S] before the loop.
# Re-init per chunk if this loop ever becomes multi-chunk. float32 keeps the
# accumulation/precision stable across the denoising schedule.
if use_motion_cache:
    dit.A = torch.zeros(B, S, device=device)
    dit._prev_noise_tokens = None
    dit._mc_phase1_count = 0

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
        is_dense = progress_id < num_dense_steps or progress_id == 10 or progress_id == 13
        selected_patches = all_patches if is_dense else None

        # MotionCache: on the first sparse step, seed the drift reference from the
        # FULL prediction two steps back (SkyReels seeds previous_e0 from the
        # warm-up forward). prev_noise_tokens is the most recent full prediction
        # (prev_guided_full); comparing it against the step-before full prediction
        # gives a non-zero drift for every token. Seeding from the scattered
        # prev_guided_noise_tokens would make the first sparse step's drift
        # exactly 0 (current == reference) and lock selection onto the first
        # frames. Falls back to a zero drift for that one step if the two-step
        # history is unavailable (e.g. a single dense warm-up step).
        if use_motion_cache and not is_dense and dit._prev_noise_tokens is None:
            dit._prev_noise_tokens = prev_guided_full_prev

        # --- Positive (conditional) forward ---
        noise_posi, noise_posi_tokens = dit.forward(
            x=latents,
            timestep=t,
            context=ctx_posi,
            kv_cache=kv_cache_posi,
            ctx_kv_cache=ctx_kv_cache_posi,
            skip_list=skip_list,
            skip_k=skip_k,
            selected_patches=selected_patches,
            ratio=ratio,
            dumb_update=dumb_update,
            enable_debug_masks=enable_viz,
            use_heuristics=use_heuristics,
            prev_noise_tokens=(
                prev_guided_full if use_motion_cache else prev_guided_noise_tokens
            ),
            dumb_noise_tokens=prev_posi_noise_tokens,
            return_noise_tokens=True,
        )

        # Free transient memory from posi forward before running nega forward.
        # The posi KV cache must persist for the next step, but intermediate
        # activations from the posi forward can be released.
        torch.cuda.empty_cache()

        # --- Classifier-free guidance ---
        if cfg_scale != 1.0:
            # The negative branch must process the SAME tokens as the positive
            # branch. On sparse steps (selected_patches is None) pass the posi
            # selection explicitly; otherwise forward() would re-derive it from
            # an independently selected noise map and double-update the skip record.
            nega_selected = (
                dit.get_last_selected_patches()
                if selected_patches is None else selected_patches
            )
            noise_nega, noise_nega_tokens = dit.forward(
                x=latents,
                timestep=t,
                context=ctx_nega,
                kv_cache=kv_cache_nega,
                ctx_kv_cache=ctx_kv_cache_nega,
                skip_list=skip_list,
                skip_k=skip_k,
                selected_patches=nega_selected,
                ratio=ratio,
                dumb_update=dumb_update,
                enable_debug_masks=False,   # only record masks for positive branch
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

        # MotionCache: zero the accumulator at the tokens that just ran a forward
        # pass (paper Eq 13: "upon selection ... its accumulator A[p] is reset
        # to 0"). ONCE per step, after the positive branch; the negative branch
        # reuses the positive selection and must not double-update.
        if use_motion_cache:
            dit.reset_motion_accumulator(active_patches)

        # The caches are RAS state, not per-branch state: scatter only the
        # active predictions into each retained cache so inactive entries keep
        # their prior values. `prev_guided_noise_tokens` is the CFG combo of the
        # two condition caches and drives the next step's selection.
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
        # Rolling history of FULL predictions for the MotionCache drift: keep
        # the last two steps' complete guided outputs (incl. dumb-updated
        # inactive tokens) so the drift compares genuinely consecutive frames.
        prev_guided_full_prev = prev_guided_full
        prev_guided_full = guided_noise_tokens.detach()

        # --- Scheduler step ---
        latents = scheduler.step(
            noise_pred,
            scheduler.timesteps[progress_id],
            latents,
        )


# ═══════════════════════════════════════════════════════════════
# Decode
# ═══════════════════════════════════════════════════════════════

# ── Free RAS memory before VAE decode ──────────────────────────
# The self-attn KV caches are the single biggest RAS VRAM cost (~[1, S, dim]
# bf16 x K/V x num_layers x 2 CFG branches ≈ ~12 GB at 480x832/81f). Nothing
# after the loop needs them, the noise caches, the starvation records, or the
# DiT itself — only `latents` and the saved masks. Releasing these keeps the
# decode phase light enough to finish on a limited-VRAM card.
del kv_cache_posi, ctx_kv_cache_posi, kv_cache_nega, ctx_kv_cache_nega
del skip_list, skip_k, all_patches
del prev_posi_noise_tokens, prev_nega_noise_tokens, prev_guided_noise_tokens
del prev_guided_full, prev_guided_full_prev
dit.to("cpu")
torch.cuda.empty_cache()
torch.cuda.synchronize()

print("Decoding video...")
vae.to(device)
video = vae.batched_tiled_decode(
    latents, device=device,
    tile_size=(30, 52), tile_stride=(15, 26),
    tile_batch_size=tile_batch_size,
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

    # Save selection masks (numpy) for external visualization.
    # viz_mask_mode controls how the temporal dimension is kept:
    #   "frame_avg"  → one [h, w] map per step (averaged over frames)   [default]
    #   "per_frame"  → one [h, w] map per (step, frame)                 (frame k)
    #   "full_grid"  → one [f, h, w] grid per step (all frames)
    import os
    os.makedirs(viz_output_dir, exist_ok=True)
    for idx, (t_val, mask, (f, h, w)) in enumerate(masks):
        grid = selection_mask_to_grid(mask, (f, h, w))   # [1, 1, f, h, w]
        grid_f = grid[0, 0]                              # [f, h, w] in patch space
        if viz_mask_mode == "frame_avg":
            # Average over frames for a summary frame mask
            frame_mask = grid_f.mean(dim=0).cpu().numpy()  # [h, w]
            np.save(f"{viz_output_dir}/mask_step_{idx:02d}_t_{t_val:.0f}.npy", frame_mask)
        elif viz_mask_mode == "per_frame":
            for k in range(f):
                fm = grid_f[k].cpu().numpy()              # [h, w] for frame k
                np.save(f"{viz_output_dir}/mask_step_{idx:02d}_f_{k:02d}_t_{t_val:.0f}.npy", fm)
        elif viz_mask_mode == "full_grid":
            np.save(f"{viz_output_dir}/mask_step_{idx:02d}_f_all_t_{t_val:.0f}.npy",
                    grid_f.cpu().numpy())                 # [f, h, w]
        else:
            raise ValueError(f"Unknown viz_mask_mode: {viz_mask_mode!r}")

    print(f"  Mask arrays saved to: {viz_output_dir}/")
    print(f"  To visualize as heatmap overlay, upsample each {h}×{w} mask to {height}×{width}")
