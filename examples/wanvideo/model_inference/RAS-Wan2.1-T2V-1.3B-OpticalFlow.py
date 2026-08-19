"""
Flow-guided RAS (Region-Adaptive Sampling) Inference Script for Wan2.1-T2V-1.3B.

Two-pass experiment. Pass 1 generates a complete dense reference video with ordinary
full-Wan CFG denoising (``kv_cache=None``) from a fixed seeded initial latent, and decodes
it to ``wan_flow_reference.mp4``. Optical flow is then estimated with pretrained RAFT-Small
(torchvision, FP32, eval/inference) over every adjacent decoded RGB frame pair. Per-pixel
flow magnitude (sqrt(u^2 + v^2)) is average-pooled to Wan's DiT patch-token grid,
aggregated temporally to the latent-frame grid (Wan's causal decoder has
T_RGB = 4 * T_latent - 3), and flattened to [B, S] in the same frame-major order as
``WanModel.patchify``.

Pass 2 restarts from an exact clone of the same initial noise and regenerates with RAS.
Sparse selection is driven by the STATIC per-token flow magnitudes, modulated by dynamic
starvation counts:

    score_i = max(flow_magnitude_i, 1e-6) * exp(starvation_scale * starvation_count_i)

The result is decoded to ``ras_flow_ranked.mp4`` for side-by-side comparison with the
dense reference. Flow is estimated ONCE from the fully generated dense reference and held
fixed during the RAS pass; only the starvation counts evolve.

Usage:
    python RAS-Wan2.1-T2V-1.3B-OpticalFlow.py [--smoke] [--ratio 0.25]
           [--num_dense_steps 10] [--starvation_scale 1.0] [--raft_microbatch 4]
           [--dense_output wan_flow_reference.mp4] [--ras_output ras_flow_ranked.mp4]
           [--enable_viz/--no-enable_viz] [--viz_mask_mode frame_avg|per_frame|full_grid]
           [--viz_output_dir ras_masks_flow]

    With ``--enable_viz`` (default), per-step binary selection masks are saved as
    ``.npy`` arrays exactly like RAS-Wan2.1-T2V-1.3B.py (via ``get_selection_masks``
    + ``selection_mask_to_grid``), into ``--viz_output_dir``.

RAFT pretrained weights (Raft_Small_Weights.DEFAULT, ~4 MB) download on first use from
download.pytorch.org. The module-level helper functions are importable for unit tests; no
models are loaded at import time (see the ``__main__`` guard).
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import (
    set_to_torch_norm,
    update_noise_cache,
    selection_mask_to_grid,
    FLASH_ATTN_3_AVAILABLE,
    FLASH_ATTN_2_AVAILABLE,
    SAGE_ATTN_AVAILABLE,
)

try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    RAFT_AVAILABLE = True
except ImportError:  # pragma: no cover - env without torchvision
    RAFT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# Configuration (module-level defaults; override via CLI)
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

# Flow-guided RAS
ratio = 0.25             # fraction of tokens processed per sparse step
num_dense_steps = 3     # initial steps with full updates to warm KV caches
starvation_scale = 1.0   # k in exp(k * starvation_count); larger = faster static-region recovery
raft_microbatch = 4      # adjacent frame pairs per RAFT forward call

# Output
dense_output_path = "wan_flow_reference.mp4"
ras_output_path = "ras_flow_ranked.mp4"

# VAE decode tiling (latent-space tile dims for the full-res run)
vae_tiled = True
vae_tile_size = (30, 52)
vae_tile_stride = (15, 26)

# RAS dumb update mode (must match the non-flow baseline script)
DUMB_UPDATE = "Previous"

# Selection-mask saving (mirrors RAS-Wan2.1-T2V-1.3B.py)
enable_viz = True           # store per-step binary selection masks
viz_mask_mode = "per_frame"  # how selection masks are saved:
                            #   "frame_avg"  → one [h, w] map per step (averaged over frames)
                            #   "per_frame"  → one [h, w] map per (step, frame)   mask_step_XX_f_KK_t_...
                            #   "full_grid"  → one [f, h, w] grid per step        mask_step_XX_f_all_t_...
viz_output_dir = "ras_masks_flow"  # directory for mask arrays


# ═══════════════════════════════════════════════════════════════
# Pure helpers (CPU-safe, importable for unit tests)
# ═══════════════════════════════════════════════════════════════

def spatial_pool_to_patch_grid(mag: torch.Tensor, patch_grid_hw: tuple) -> torch.Tensor:
    """Average-pool per-pixel flow magnitude down to the DiT patch grid.

    Args:
        mag: [N, H, W] per-pixel motion magnitudes (sqrt(u^2 + v^2)).
        patch_grid_hw: (patch_h, patch_w) — Wan's patched grid after the VAE's 8x
            spatial compression and the DiT's patch_size[1:] compression.
    Returns:
        [N, patch_h, patch_w] mean magnitude per patch. Adaptive pooling makes this
        robust to the exact H/W vs. patch-grid divisibility.
    """
    pooled = F.adaptive_avg_pool2d(mag.unsqueeze(1), patch_grid_hw)  # [N, 1, ph, pw]
    return pooled.squeeze(1)


def temporal_group_flow(flow_grid: torch.Tensor, num_latent_frames: int) -> torch.Tensor:
    """Aggregate per-pair flow to the latent-frame grid using Wan's causal structure.

    Wan's VAE decoder is causal with T_RGB = 4 * T_latent - 3, so the first latent
    frame only covers the first RGB frame pair and every subsequent latent frame
    covers four adjacent pairs:

        latent frame 0  <- flow pair 0
        latent frame k  <- flow pairs [4k-3, 4k]      (k >= 1)

    The final partial group is averaged normally (mean over whatever pairs remain).

    Args:
        flow_grid: [num_pairs, patch_h, patch_w] spatial-pooled magnitude per pair.
        num_latent_frames: T_latent.
    Returns:
        [num_latent_frames, patch_h, patch_w] magnitude per latent frame.
    """
    num_pairs = flow_grid.shape[0]
    if num_pairs == 0:
        return torch.zeros(
            num_latent_frames, *flow_grid.shape[1:], dtype=flow_grid.dtype
        )
    # Latent frame 0 gets the very first flow pair.
    groups = [flow_grid[0:1].mean(dim=0, keepdim=True)]
    start = 1
    for _ in range(1, num_latent_frames):
        end = min(start + 4, num_pairs)
        if start < end:
            groups.append(flow_grid[start:end].mean(dim=0, keepdim=True))
        else:
            # No pairs left for this latent frame (degenerate short video).
            groups.append(torch.zeros_like(flow_grid[0:1]))
        start = end
    return torch.cat(groups, dim=0)


def flow_grid_to_b_s(latent_grid: torch.Tensor) -> torch.Tensor:
    """Flatten [T_latent, patch_h, patch_w] to [1, S] in frame-major order.

    Matches ``WanModel.patchify``'s token ordering (``b (f h w) d``): f-major, then
    h, then w.
    """
    return latent_grid.reshape(1, -1)


def estimate_flow_magnitudes(rgb_frames, latent_frames, patch_grid_hw, device,
                             raft_model, raft_transforms, raft_microbatch=4):
    """Estimate per-token flow magnitudes from decoded RGB frames via RAFT.

    Args:
        rgb_frames: list of numpy arrays [H, W, C] float32 in [0, 1].
        latent_frames: T_latent (for the causal temporal grouping).
        patch_grid_hw: (patch_h, patch_w) DiT patch grid.
        device: torch device for the RAFT model.
        raft_model: pretrained torchvision ``raft_small`` (FP32, eval mode).
        raft_transforms: ``Raft_Small_Weights.DEFAULT.transforms()`` — maps [0,1] to
            [-1,1] (the model itself does not normalize its inputs).
        raft_microbatch: adjacent frame pairs per RAFT forward call.
    Returns:
        [1, S] float32 tensor of per-token flow magnitudes (CPU).
    """
    num_pairs = len(rgb_frames) - 1
    if num_pairs <= 0:
        raise ValueError(f"need at least 2 RGB frames to estimate flow, got {len(rgb_frames)}")
    pooled = []
    first = True
    with torch.inference_mode():
        for start in range(0, num_pairs, raft_microbatch):
            end = min(start + raft_microbatch, num_pairs)
            img1 = torch.stack([
                torch.from_numpy(rgb_frames[i]).permute(2, 0, 1).float()
                for i in range(start, end)
            ])  # [N, 3, H, W] in [0, 1]
            img2 = torch.stack([
                torch.from_numpy(rgb_frames[i + 1]).permute(2, 0, 1).float()
                for i in range(start, end)
            ])
            img1, img2 = raft_transforms(img1, img2)  # [0,1] -> [-1,1], contiguous
            flows = raft_model(img1.to(device), img2.to(device))
            flow = flows[-1].float()                  # [N, 2, H, W] (final prediction)
            if first:
                print(f"    RAFT flow field: {tuple(flow.shape)} "
                      f"(final of {len(flows)} updates)")
                first = False
            mag = flow.square().sum(dim=1).sqrt()     # [N, H, W]
            pooled.append(spatial_pool_to_patch_grid(mag, patch_grid_hw).cpu())
            del img1, img2, flows, flow, mag
    flow_grid = torch.cat(pooled, dim=0)              # [num_pairs, patch_h, patch_w]
    latent_grid = temporal_group_flow(flow_grid, latent_frames)
    return flow_grid_to_b_s(latent_grid)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Flow-guided RAS regeneration for Wan2.1-T2V-1.3B")
    parser.add_argument("--ratio", type=float, default=ratio, help="fraction of tokens processed per sparse step")
    parser.add_argument("--num_dense_steps", type=int, default=num_dense_steps, help="initial dense warm-up steps")
    parser.add_argument("--starvation_scale", type=float, default=starvation_scale, help="k in exp(k * starvation_count)")
    parser.add_argument("--raft_microbatch", type=int, default=raft_microbatch, help="adjacent frame pairs per RAFT call")
    parser.add_argument("--num_inference_steps", type=int, default=num_inference_steps)
    parser.add_argument("--cfg_scale", type=float, default=cfg_scale)
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument("--num_frames", type=int, default=num_frames)
    parser.add_argument("--height", type=int, default=height)
    parser.add_argument("--width", type=int, default=width)
    parser.add_argument("--dense_output", type=str, default=dense_output_path)
    parser.add_argument("--ras_output", type=str, default=ras_output_path)
    parser.add_argument("--enable_viz", action=argparse.BooleanOptionalAction, default=enable_viz,
                        help="store per-step binary selection masks (--no-enable_viz to disable)")
    parser.add_argument("--viz_mask_mode", type=str, default=viz_mask_mode,
                        choices=["frame_avg", "per_frame", "full_grid"])
    parser.add_argument("--viz_output_dir", type=str, default=viz_output_dir)
    parser.add_argument("--smoke", action="store_true",
                        help="reduced settings for a quick GPU smoke test (17 frames, 64x128, 3 steps)")
    args = parser.parse_args()
    if args.smoke:
        # Tiny latents: disable tiled decode and shrink everything else.
        args.num_frames = 17
        args.height = 64
        args.width = 128
        args.num_inference_steps = 3
        args.num_dense_steps = 1
        args.vae_tiled = False
    else:
        args.vae_tiled = vae_tiled
    return args


# ═══════════════════════════════════════════════════════════════
# Model / pipeline setup (mirrors RAS-Wan2.1-T2V-1.3B.py)
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


def _cuda_sync():
    """Release GPU memory and sync, no-op when running on CPU."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if not RAFT_AVAILABLE:
        raise ImportError("torchvision.models.optical_flow is required; install torchvision")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print("Loading model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
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
    print(f"  Patch size: {dit.patch_size}")
    print(f"  Attention: FA3={FLASH_ATTN_3_AVAILABLE}, FA2={FLASH_ATTN_2_AVAILABLE}, Sage={SAGE_ATTN_AVAILABLE}")

    set_to_torch_norm([dit])
    dit.eval()

    print("Encoding prompts...")
    ctx_posi = encode_prompt(pipe, prompt)       # [1, seq_len, text_dim]
    ctx_nega = encode_prompt(pipe, negative_prompt)

    # ── Initial latents: generated ONCE, cloned for both passes ──
    print("Initializing latents...")
    z_dim = vae.model.z_dim
    latent_frames = (args.num_frames - 1) // 4 + 1     # temporal compression (causal)
    latent_h = args.height // vae.upsampling_factor     # spatial compression
    latent_w = args.width // vae.upsampling_factor
    patch_h = latent_h // dit.patch_size[1]             # DiT patch grid
    patch_w = latent_w // dit.patch_size[2]

    shape = (1, z_dim, latent_frames, latent_h, latent_w)
    initial_latents = pipe.generate_noise(
        shape, seed=args.seed, rand_device="cpu",
    ).to(dtype=dtype, device=device)
    print(f"  Latent shape: {tuple(initial_latents.shape)}  "
          f"(patch grid {latent_frames}x{patch_h}x{patch_w})")

    scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    timesteps = scheduler.timesteps
    print(f"  Timesteps: {len(timesteps)} steps, range [{timesteps[0]:.1f}, {timesteps[-1]:.1f}]")

    # Free GPU memory: T5 text encoder is done; VAE only needed for decode.
    for model_attr, name in [
        (pipe.text_encoder, "text_encoder"),
        (pipe.vae, "vae"),
        (getattr(pipe, "image_encoder", None), "image_encoder"),
        (getattr(pipe, "motion_controller", None), "motion_controller"),
    ]:
        if model_attr is not None:
            model_attr.to("cpu")
            print(f"  Moved {name} to CPU")
    _cuda_sync()

    # ═══════════════════════════════════════════════════════════
    # Pass 1 — dense reference (ordinary full-Wan CFG, kv_cache=None)
    # ═══════════════════════════════════════════════════════════
    latents = initial_latents.clone()
    print(f"\nDense reference pass ({len(timesteps)} full-Wan steps)...")
    with torch.inference_mode():
        for progress_id, timestep in enumerate(tqdm(timesteps, desc="Dense reference")):
            t = timestep.unsqueeze(0).to(dtype=dtype, device=device)
            noise_posi = dit.forward(x=latents, timestep=t, context=ctx_posi)
            noise_nega = dit.forward(x=latents, timestep=t, context=ctx_nega)
            noise_pred = noise_nega + args.cfg_scale * (noise_posi - noise_nega)
            latents = scheduler.step(noise_pred, timesteps[progress_id], latents)

    print("Decoding reference video...")
    vae.to(device)
    video_tensor = vae.decode(
        latents, device=device,
        tiled=args.vae_tiled, tile_size=vae_tile_size, tile_stride=vae_tile_stride,
    )
    video_frames = pipe.vae_output_to_video(video_tensor)          # list of PIL Images
    save_video(video_frames, args.dense_output, fps=15, quality=5)
    print(f"Dense reference saved: {args.dense_output}")

    # Decoded RGB frames as float32 [0, 1] numpy arrays for RAFT.
    rgb_frames = [np.asarray(f, dtype=np.float32) / 255.0 for f in video_frames]
    print(f"  Decoded {len(rgb_frames)} RGB frames, each {rgb_frames[0].shape}")

    # ═══════════════════════════════════════════════════════════
    # RAFT-Small flow estimation (FP32 / eval / inference_mode)
    # ═══════════════════════════════════════════════════════════
    vae.to("cpu")
    _cuda_sync()

    print("Loading RAFT-Small (weights download on first use)...")
    raft_model = raft_small(weights=Raft_Small_Weights.DEFAULT)
    raft_model = raft_model.to(device=device).eval().float()
    raft_transforms = Raft_Small_Weights.DEFAULT.transforms()

    flow_magnitudes = estimate_flow_magnitudes(
        rgb_frames, latent_frames, (patch_h, patch_w), device,
        raft_model, raft_transforms, args.raft_microbatch,
    )
    print(f"  flow_magnitudes: {tuple(flow_magnitudes.shape)}  "
          f"(max={flow_magnitudes.max().item():.3f}, mean={flow_magnitudes.mean().item():.4f})")
    flow_magnitudes = flow_magnitudes.to(device=device)

    # Release decoded reference frames, VAE working memory, and RAFT temporaries
    # before the RAS regeneration pass.
    del video_tensor, video_frames, rgb_frames, raft_model
    _cuda_sync()

    # ═══════════════════════════════════════════════════════════
    # Pass 2 — flow-guided RAS regeneration
    # ═══════════════════════════════════════════════════════════
    B = initial_latents.shape[0]
    S = latent_frames * patch_h * patch_w
    num_layers = len(dit.blocks)

    # Fresh RAS state: KV caches, noise caches, skip counters, debug masks.
    kv_cache_posi = [{} for _ in range(num_layers)]
    ctx_kv_cache_posi = [{} for _ in range(num_layers)]
    kv_cache_nega = [{} for _ in range(num_layers)]
    ctx_kv_cache_nega = [{} for _ in range(num_layers)]
    skip_list = torch.zeros(B, S, device=device)
    skip_k = torch.zeros(B, S, device=device)
    prev_posi_noise_tokens = None
    prev_nega_noise_tokens = None
    prev_guided_noise_tokens = None
    all_patches = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    dit.clear_selection_masks()

    latents = initial_latents.clone()
    assert torch.equal(latents, initial_latents), \
        "RAS pass must start from an identical clone of the initial latents"
    print(f"\nFlow-guided RAS pass ({len(timesteps)} steps, ratio={args.ratio}, "
          f"dense warm-up={args.num_dense_steps}, starvation_scale={args.starvation_scale})...")

    selected_counts = []
    with torch.inference_mode():
        for progress_id, timestep in enumerate(tqdm(timesteps, desc="RAS (flow-guided)")):
            t = timestep.unsqueeze(0).to(dtype=dtype, device=device)

            # Dense warm-up steps process ALL tokens to seed per-layer KV caches;
            # sparse steps let flow-guided select_region pick the tokens.
            is_dense = progress_id < args.num_dense_steps or progress_id == 10 or progress_id == 13
            selected_patches = all_patches if is_dense else None

            # --- Positive (conditional) forward: flow-guided selection ---
            noise_posi, noise_posi_tokens = dit.forward(
                x=latents,
                timestep=t,
                context=ctx_posi,
                kv_cache=kv_cache_posi,
                ctx_kv_cache=ctx_kv_cache_posi,
                skip_list=skip_list,
                skip_k=skip_k,
                selected_patches=selected_patches,
                ratio=args.ratio,
                dumb_update=DUMB_UPDATE,
                enable_debug_masks=args.enable_viz,   # record per-step masks only once (positive branch)
                use_heuristics="flow",
                prev_noise_tokens=prev_guided_noise_tokens,
                dumb_noise_tokens=prev_posi_noise_tokens,
                flow_magnitudes=flow_magnitudes,
                starvation_scale=args.starvation_scale,
                return_noise_tokens=True,
            )
            _cuda_sync()

            # --- Classifier-free guidance ---
            if args.cfg_scale != 1.0:
                # Negative branch reuses the positive branch's EXACT selection so the
                # starvation record is updated only once (in the positive branch).
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
                    ratio=args.ratio,
                    dumb_update=DUMB_UPDATE,
                    enable_debug_masks=False,   # only record masks for positive branch
                    prev_noise_tokens=prev_guided_noise_tokens,
                    dumb_noise_tokens=prev_nega_noise_tokens,
                    return_noise_tokens=True,
                )
                noise_pred = noise_nega + args.cfg_scale * (noise_posi - noise_nega)
                guided_noise_tokens = noise_nega_tokens + args.cfg_scale * (
                    noise_posi_tokens - noise_nega_tokens
                )
                active_patches = nega_selected
            else:
                noise_pred = noise_posi
                guided_noise_tokens = noise_posi_tokens
                active_patches = dit.get_last_selected_patches()

            # Per-condition noise caches updated once per step (RAS state, not
            # per-branch state); flow, not previous noise, drives selection.
            prev_posi_noise_tokens = update_noise_cache(
                prev_posi_noise_tokens, active_patches, noise_posi_tokens,
            )
            if args.cfg_scale != 1.0:
                prev_nega_noise_tokens = update_noise_cache(
                    prev_nega_noise_tokens, active_patches, noise_nega_tokens,
                )
            prev_guided_noise_tokens = update_noise_cache(
                prev_guided_noise_tokens, active_patches, guided_noise_tokens,
            )

            selected_counts.append(int(active_patches.shape[1]))
            latents = scheduler.step(noise_pred, timesteps[progress_id], latents)

    expected_active = math.ceil(args.ratio * S)
    print(f"  Selected-token counts per step: {selected_counts}")
    print(f"  Sparse step active count: {selected_counts[-1]} of {S} "
          f"(expected ceil({args.ratio} x {S}) = {expected_active})")

    print("Decoding flow-guided RAS video...")
    vae.to(device)
    video_tensor = vae.decode(
        latents, device=device,
        tiled=args.vae_tiled, tile_size=vae_tile_size, tile_stride=vae_tile_stride,
    )
    video_frames = pipe.vae_output_to_video(video_tensor)
    save_video(video_frames, args.ras_output, fps=15, quality=5)
    print(f"Flow-guided RAS video saved: {args.ras_output}")

    # ═══════════════════════════════════════════════════════════
    # Save binary selection masks (mirrors RAS-Wan2.1-T2V-1.3B.py)
    # ═══════════════════════════════════════════════════════════

    if args.enable_viz:
        print("\nSaving selection masks...")
        masks = dit.get_selection_masks()
        print(f"  Recorded {len(masks)} per-step masks")

        # Each mask entry: (timestep_value, mask_tensor [B, S], grid_size (f, h, w))
        for idx, (t_val, mask, (f, h, w)) in enumerate(masks):
            # Convert [B, S] bool mask → [B, 1, f, h, w] float spatial grid
            grid = selection_mask_to_grid(mask, (f, h, w))
            frac_selected = mask.float().mean().item()
            print(f"  Step {idx:2d} (t={t_val:6.1f}): "
                  f"{frac_selected*100:4.1f}% tokens selected  "
                  f"grid shape={list(grid.shape)}  "
                  f"grid_size=({f},{h},{w})")

        os.makedirs(args.viz_output_dir, exist_ok=True)
        for idx, (t_val, mask, (f, h, w)) in enumerate(masks):
            grid = selection_mask_to_grid(mask, (f, h, w))   # [1, 1, f, h, w]
            grid_f = grid[0, 0]                              # [f, h, w] in patch space
            if args.viz_mask_mode == "frame_avg":
                # Average over frames for a summary frame mask
                frame_mask = grid_f.mean(dim=0).cpu().numpy()  # [h, w]
                np.save(f"{args.viz_output_dir}/mask_step_{idx:02d}_t_{t_val:.0f}.npy", frame_mask)
            elif args.viz_mask_mode == "per_frame":
                for k in range(f):
                    fm = grid_f[k].cpu().numpy()              # [h, w] for frame k
                    np.save(f"{args.viz_output_dir}/mask_step_{idx:02d}_f_{k:02d}_t_{t_val:.0f}.npy", fm)
            elif args.viz_mask_mode == "full_grid":
                np.save(f"{args.viz_output_dir}/mask_step_{idx:02d}_f_all_t_{t_val:.0f}.npy",
                        grid_f.cpu().numpy())                 # [f, h, w]
            else:
                raise ValueError(f"Unknown viz_mask_mode: {args.viz_mask_mode!r}")

        print(f"  Mask arrays saved to: {args.viz_output_dir}/")
        print(f"  To visualize as heatmap overlay, upsample each {h}×{w} mask to {height}×{width}")

    print("\nDone. Compare:")
    print(f"  dense reference : {args.dense_output}")
    print(f"  flow-guided RAS : {args.ras_output}")


if __name__ == "__main__":
    main()
