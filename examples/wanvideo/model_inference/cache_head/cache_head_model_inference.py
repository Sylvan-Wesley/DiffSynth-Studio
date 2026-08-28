"""
Hybrid Wan + CacheHead inference runner.

Runs the 15-step schedule with the full Wan model (positive/negative CFG) only
at the anchor steps and the lightweight CacheHead at the other ten steps:

    full step:  run full Wan CFG, refresh prev_guided_noise_tokens
    head step:  v_tokens = prev_guided_noise_tokens
                        + CacheHead(prev_guided_noise_tokens, t)
                noise_pred = unpatchify(v_tokens)   (Wan's exact inverse)

All predictions go through the unchanged Wan scheduler step.  No RAS, sparse
KV, MotionCache, token selection, or selector training.

Modes:
    hybrid  full anchors + CacheHead head steps (checkpoint, or zero-init = carry)
    full    15 full-Wan steps (baseline trajectory)
    carry   zero-init head so head steps exactly carry the previous guided tokens

The latent trajectory (init + after each scheduler update, 16 states for the
15-step schedule) is recorded for PCA trajectory-difference evaluation.

The pure sampling loop lives in ``HybridSampler`` / ``full_step`` /
``head_step`` and is CPU-testable with a fake dit and scheduler; only
``main()`` loads the real Wan model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cache_head_model import (
    CacheHead,
    CacheHeadConfig,
    CacheHeadSchedule,
    load_cache_head,
    unpatchify_tokens,
)

# Heavy imports are deferred to main() so this module stays importable on a
# machine without the Wan dependencies installed.


# ═══════════════════════════════════════════════════════════════
# Per-step primitives (no torch.no_grad: callers control grad context)
# ═══════════════════════════════════════════════════════════════

def full_step(dit, latents, timestep, ctx_posi, ctx_nega, cfg_scale, **dit_kwargs):
    """Full Wan denoising step with positive/negative CFG.

    Returns (noise_pred [B,C,F,H,W], guided_noise_tokens [B,S,64]).
    """
    noise_posi, posi_tokens = dit(
        x=latents, timestep=timestep, context=ctx_posi, return_noise_tokens=True, **dit_kwargs
    )
    noise_nega, nega_tokens = dit(
        x=latents, timestep=timestep, context=ctx_nega, return_noise_tokens=True, **dit_kwargs
    )
    noise_pred = noise_nega + cfg_scale * (noise_posi - noise_nega)
    guided_tokens = nega_tokens + cfg_scale * (posi_tokens - nega_tokens)
    return noise_pred, guided_tokens


def head_step(head, timestep, prev_guided_tokens, grid, patch_size):
    """CacheHead denoising step: residual on the nearest preceding guided
    tokens, unpatchified to the latent velocity.

    Returns (noise_pred [B,C,F,H,W], v_tokens [B,S,64]).
    """
    residual = head(prev_guided_tokens, timestep, grid)
    v_tokens = prev_guided_tokens + residual
    noise_pred = unpatchify_tokens(v_tokens, grid, patch_size)
    return noise_pred, v_tokens


# ═══════════════════════════════════════════════════════════════
# Hybrid sampler
# ═══════════════════════════════════════════════════════════════

class HybridSampler:
    """Drives a Wan scheduler through the CacheHead schedule.

    ``dit`` must expose ``forward(x, timestep, context, return_noise_tokens=True)``
    returning ``(noise_pred, noise_tokens)``; ``scheduler`` must match
    ``FlowMatchScheduler``'s ``timesteps`` and ``step(model_output, timestep, sample)``.
    """

    def __init__(self, dit, scheduler, head, schedule, cfg_scale, patch_size, grid):
        self.dit = dit
        self.scheduler = scheduler
        self.head = head
        self.schedule = schedule
        self.cfg_scale = cfg_scale
        self.patch_size = patch_size
        self.grid = grid

    @torch.no_grad()
    def sample(self, latents, ctx_posi, ctx_nega):
        """Run one rollout.  Returns (final_latents, states, stats) where
        ``states`` has length num_inference_steps + 1 (init + after each update)
        and ``stats`` counts full/head calls."""
        states = [latents.detach().float().cpu()]
        prev_guided = None
        stats = {"full_calls": 0, "head_calls": 0}
        timesteps = self.scheduler.timesteps
        for progress_id, timestep in enumerate(timesteps):
            t = timestep.reshape(1).to(device=latents.device, dtype=latents.dtype)
            if self.schedule.is_full_step(progress_id):
                noise_pred, prev_guided = full_step(
                    self.dit, latents, t, ctx_posi, ctx_nega, self.cfg_scale
                )
                stats["full_calls"] += 1
            else:
                if prev_guided is None:
                    raise RuntimeError(
                        f"head step at progress {progress_id} before any full step; "
                        f"invalid schedule {self.schedule}"
                    )
                noise_pred, prev_guided = head_step(
                    self.head, t, prev_guided, self.grid, self.patch_size
                )
                stats["head_calls"] += 1
            latents = self.scheduler.step(noise_pred, timestep, latents)
            states.append(latents.detach().float().cpu())
        return latents, states, stats


# ═══════════════════════════════════════════════════════════════
# Wan setup + CLI (requires the Wan pipeline installed)
# ═══════════════════════════════════════════════════════════════

def _build_schedule(mode: str, num_steps: int, config: CacheHeadConfig | None) -> CacheHeadSchedule:
    if mode == "full":
        return CacheHeadSchedule(num_inference_steps=num_steps, full_step_indices=tuple(range(1, num_steps + 1)))
    if config is not None:
        return config.schedule
    return CacheHeadSchedule(num_inference_steps=num_steps)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _save_trajectory(npz_path, states, scheduler, method, prompt_id, seed):
    import numpy as np

    timesteps = scheduler.timesteps.detach().cpu().numpy()
    sigmas = scheduler.sigmas.detach().cpu().numpy()
    states_np = np.stack([s.squeeze(0).numpy() for s in states])  # [steps, C, F, H, W]
    np.savez_compressed(
        npz_path,
        states=states_np,
        timesteps=timesteps,
        sigmas=sigmas,
        method=np.asarray([method]),
        prompt_id=np.asarray([prompt_id]),
        seed=np.asarray([seed], dtype=np.int64),
        step_indices=np.arange(states_np.shape[0]),
    )
    return npz_path


def run_pipeline(args) -> None:
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.models.wan_video_dit import set_to_torch_norm
    from diffsynth.utils.data import save_video

    device = _detect_device()
    dtype = torch.bfloat16 if getattr(args, "dtype", "bf16") == "bf16" else torch.float16
    print(f"Device: {device}, dtype: {dtype}")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="google/umt5-xxl/"),
    )
    dit = pipe.dit
    scheduler = pipe.scheduler
    set_to_torch_norm([dit])
    dit.eval()
    dit.requires_grad_(False)

    # Head + config.
    config = None
    if args.checkpoint and Path(args.checkpoint).is_file():
        head, config = load_cache_head(args.checkpoint, device=device, dtype=dtype)
        print(f"Loaded CacheHead from {args.checkpoint} (cfg={config.cfg_scale})")
    else:
        config = CacheHeadConfig(model_id=args.model_id, cfg_scale=args.cfg)
        head = CacheHead(config).to(device=device, dtype=dtype).eval()
        print("No checkpoint: using zero-initialized head (carry_previous)")

    if args.mode == "carry":
        head = CacheHead(CacheHeadConfig(model_id=args.model_id, cfg_scale=args.cfg)).to(
            device=device, dtype=dtype
        ).eval()
        print("Mode=carry: forcing zero-init head (carry_previous)")

    schedule = _build_schedule(args.mode, args.num_steps, config)
    print(f"Schedule: {schedule.num_inference_steps} steps, "
          f"full={schedule.full_step_indices}, head={schedule.head_step_indices}")
    cfg_scale = config.cfg_scale if args.mode == "hybrid" and config is not None else args.cfg
    scheduler.set_timesteps(schedule.num_inference_steps, denoising_strength=1.0, shift=5.0)

    # Shape math.
    z_dim = pipe.vae.model.z_dim
    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_h = args.height // pipe.vae.upsampling_factor
    latent_w = args.width // pipe.vae.upsampling_factor
    shape = (1, z_dim, latent_frames, latent_h, latent_w)
    grid = (latent_frames, latent_h // dit.patch_size[1], latent_w // dit.patch_size[2])
    S = grid[0] * grid[1] * grid[2]
    print(f"Latent {shape}, token grid {grid}, S={S}")

    sampler = HybridSampler(dit, scheduler, head, schedule, cfg_scale, dit.patch_size, grid)

    def encode_prompt(text):
        ids, mask = pipe.tokenizer(text, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = pipe.text_encoder(ids, mask)
        for v in seq_lens:
            emb[:, v:] = 0
        return emb

    ctx_posi = encode_prompt(args.prompt)
    ctx_nega = encode_prompt(args.negative_prompt)

    # Offload unused models to CPU.
    for model_attr, name in [
        (pipe.text_encoder, "text_encoder"),
        (pipe.vae, "vae"),
        (getattr(pipe, "image_encoder", None), "image_encoder"),
        (getattr(pipe, "motion_controller", None), "motion_controller"),
    ]:
        if model_attr is not None:
            model_attr.to("cpu")
            print(f"  Moved {name} to CPU")
    torch.cuda.empty_cache() if device == "cuda" else None

    latents = pipe.generate_noise(shape, seed=args.seed, rand_device="cpu").to(
        dtype=dtype, device=device
    )

    torch.cuda.synchronize() if device == "cuda" else None
    t_start = time.perf_counter()
    final_latents, states, stats = sampler.sample(latents, ctx_posi, ctx_nega)
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - t_start
    print(f"Sampling done in {elapsed:.2f}s: {stats} "
          f"({elapsed / schedule.num_inference_steps * 1000:.0f} ms/step)")

    if args.trajectory:
        _save_trajectory(
            args.trajectory, states, scheduler,
            method=args.mode, prompt_id=getattr(args, "prompt_id", 0), seed=args.seed,
        )
        print(f"Trajectory saved to {args.trajectory}")

    if args.output:
        if device == "cuda":
            torch.cuda.empty_cache()
        video = pipe.vae.decode(
            final_latents.to(device=device),
            device=device,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
        )
        save_video(pipe.vae_output_to_video(video), args.output, fps=15, quality=5)
        print(f"Video saved to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Wan + CacheHead inference")
    parser.add_argument("--checkpoint", default=None, help="CacheHead checkpoint (schedule/config stored inside)")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    ))
    parser.add_argument("--num-steps", type=int, default=15)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--mode", choices=["hybrid", "full", "carry"], default="hybrid")
    parser.add_argument("--output", default=None)
    parser.add_argument("--trajectory", default=None)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    args = parser.parse_args()
    if args.prompt is None:
        raise SystemExit("--prompt is required")
    run_pipeline(args)


if __name__ == "__main__":
    main()
