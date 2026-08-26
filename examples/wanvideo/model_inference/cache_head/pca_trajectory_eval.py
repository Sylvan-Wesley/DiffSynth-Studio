"""
PCA trajectory-difference evaluation for CacheHead.

For a locked probe panel of held-out prompts x seeds, captures the latent
trajectory (init + after every scheduler update: 16 states for the 15-step
schedule) under three methods:

    full_wan        all 15 full-Wan CFG steps
    carry_previous  hybrid schedule, zero-init head (head steps carry tokens)
    <arm>           hybrid schedule, trained CacheHead

ONE shared two-dimensional PCA basis is fit jointly over all three methods'
states (never per-method), preserving the deterministic seed/sign behavior and
resumable per-run caching of ``analyze_geometric_trajectory.py``.

Outputs (per evaluated checkpoint):
    pca_trajectory.npz        method labels, prompt ids, seeds, step indices,
                              timesteps, sigmas, shared mean/components,
                              explained variance, [method,prompt,seed,step,2] coords
    pca_trajectory.png        one shared plane: full solid, carry dashed,
                              CacheHead dotted; start/final/anchor/head distinct
    trajectory_metrics.json   PCA pointwise distance to full, cumulative PCA
                              path distance, terminal PCA distance, and
                              full-dimensional latent L2/cosine/relative-drift

PCA coordinates are visual diagnostics only; go/no-go decisions use the
full-dimensional latent metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the shared-PCA helpers from analyze_geometric_trajectory.py (pure, CPU-safe).
PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
from analyze_geometric_trajectory import (  # noqa: E402
    _cache_fingerprint,
    _load_cached_run,
    _save_run_cache,
    fit_shared_pca_2d,
)

from cache_head_model import (  # noqa: E402
    CacheHead,
    CacheHeadConfig,
    CacheHeadSchedule,
    load_cache_head,
)
from cache_head_model_inference import HybridSampler  # noqa: E402


METHOD_FULL = "full_wan"
METHOD_CARRY = "carry_previous"


# ═══════════════════════════════════════════════════════════════
# Pure analysis (CPU-testable, no Wan needed)
# ═══════════════════════════════════════════════════════════════

def build_shared_pca(
    runs: dict[str, np.ndarray],
    pca_seed: int = 0,
    device: torch.device | str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Fit ONE shared 2D PCA basis jointly over all methods' states.

    ``runs`` maps method -> [P, steps, ...] latent states (P = prompts*seeds).
    Returns (fit, per_method_coords) where fit has the shared mean/components/
    explained_variance_ratio/coordinates and per_method_coords maps method ->
    [P, steps, 2] split from the single shared fit.
    """
    methods = list(runs)
    if not methods:
        raise ValueError("no runs provided")
    all_traj = np.concatenate([runs[m] for m in methods], axis=0)
    fit = fit_shared_pca_2d(all_traj, random_seed=pca_seed, device=device)
    coords = fit["coordinates"]  # [sum(P), steps, 2]
    per_method_coords: dict[str, np.ndarray] = {}
    offset = 0
    for m in methods:
        n = runs[m].shape[0]
        per_method_coords[m] = coords[offset:offset + n]
        offset += n
    return fit, per_method_coords


def _summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "final_step_mean": float(np.nanmean(values[..., -1])) if values.ndim > 1 else float(np.nanmean(values)),
    }


def trajectory_metrics(
    states: dict[str, np.ndarray],
    coords: dict[str, np.ndarray],
    full: str = METHOD_FULL,
) -> dict:
    """Full-dimensional + PCA trajectory metrics for every method vs ``full``."""
    if full not in states or full not in coords:
        raise ValueError(f"full method {full!r} missing from runs")
    full_s = states[full].reshape(states[full].shape[0], states[full].shape[1], -1)
    full_c = coords[full]
    out: dict = {}
    for m in states:
        if m == full:
            continue
        s = states[m].reshape(states[m].shape[0], states[m].shape[1], -1)
        c = coords[m]
        # PCA pointwise distance to full (per step, per run).
        pointwise_pca = np.linalg.norm(c - full_c, axis=-1)          # [P, T]
        # Cumulative PCA path distance (per run).
        cum_pca = np.sum(np.linalg.norm(np.diff(c, axis=1), axis=-1), axis=1)
        # Terminal PCA distance.
        term_pca = np.linalg.norm(c[:, -1] - full_c[:, -1], axis=-1)
        # Full-dimensional latent metrics.
        l2 = np.linalg.norm(s - full_s, axis=-1)                      # [P, T]
        denom = np.linalg.norm(full_s, axis=-1) + 1e-8
        rel = l2 / denom
        num = (s * full_s).sum(-1)
        cos = num / (np.linalg.norm(s, axis=-1) * np.linalg.norm(full_s, axis=-1) + 1e-8)
        out[m] = {
            "pca_pointwise_distance": _summarize(pointwise_pca),
            "pca_cumulative_path_distance": _summarize(cum_pca[:, None]),
            "pca_terminal_distance": _summarize(term_pca[:, None]),
            "latent_l2": _summarize(l2),
            "latent_cosine": _summarize(cos),
            "latent_relative_drift": _summarize(rel),
        }
    return out


def make_trajectory_plot(
    coords: dict[str, np.ndarray],
    explained_variance_ratio: np.ndarray,
    full_anchor_steps: list[int],
    head_steps: list[int],
    path: str,
) -> None:
    """One shared plane: full solid, carry dashed, CacheHead dotted.

    Mark start, final, full-anchor, and head steps distinctly.
    ``full_anchor_steps``/``head_steps`` are 0-indexed state indices.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    line_style = {METHOD_FULL: "-", METHOD_CARRY: "--", None: ":"}
    color = {METHOD_FULL: "tab:blue", METHOD_CARRY: "tab:orange", None: "tab:green"}
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    for m, c in coords.items():
        style = line_style.get(m, ":")
        col = color.get(m, "tab:green")
        x, y = c[..., 0], c[..., 1]
        ax.plot(x.T, y.T, style, color=col, linewidth=1.2, alpha=0.8, label=m)
        ax.scatter(x[:, 0], y[:, 0], s=60, marker="o", facecolors="none", edgecolors=col, zorder=3)
        ax.scatter(x[:, -1], y[:, -1], s=80, marker="X", color=col, zorder=3)
    # Anchor and head-step markers (from one representative run).
    rep = next(iter(coords.values()))[0]
    for s in head_steps:
        ax.scatter(rep[s, 0], rep[s, 1], s=12, marker=".", color="black", alpha=0.6, zorder=2)
    for s in full_anchor_steps:
        ax.scatter(rep[s, 0], rep[s, 1], s=28, marker="D", facecolors="none",
                   edgecolors="black", alpha=0.7, zorder=2)
    ax.set_title("CacheHead latent denoising trajectories (start ○, final ✕, anchors ◇, head ·)")
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_artifacts(
    npz_path: str,
    png_path: str,
    metrics_path: str,
    *,
    fit: dict[str, np.ndarray],
    per_method_coords: dict[str, np.ndarray],
    method_labels: list[str],
    prompt_ids: list[str],
    seeds: list[int],
    timesteps: np.ndarray,
    sigmas: np.ndarray,
    metrics: dict,
) -> None:
    """Persist the shared PCA coordinates in [method, prompt, seed, step, 2] order."""
    n_methods = len(method_labels)
    n_prompts = len(prompt_ids)
    n_seeds = len(seeds)
    n_steps = per_method_coords[method_labels[0]].shape[1]
    coords_5d = np.full((n_methods, n_prompts, n_seeds, n_steps, 2), np.nan)
    for mi, m in enumerate(method_labels):
        arr = per_method_coords[m]  # [P, steps, 2], P ordered prompt-major then seed
        coords_5d[mi] = arr.reshape(n_prompts, n_seeds, n_steps, 2)

    np.savez_compressed(
        npz_path,
        method_labels=np.asarray(method_labels),
        prompt_ids=np.asarray(prompt_ids),
        seeds=np.asarray(seeds, dtype=np.int64),
        step_indices=np.arange(n_steps),
        timesteps=timesteps,
        sigmas=sigmas,
        pca_mean=fit["mean"],
        pca_components=fit["components"],
        explained_variance_ratio=fit["explained_variance_ratio"],
        coordinates=coords_5d,
        latent_shape=np.asarray(per_method_coords[method_labels[0]].shape[2:], dtype=np.int64),
    )
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)


# ═══════════════════════════════════════════════════════════════
# Rollout capture (requires the Wan pipeline)
# ═══════════════════════════════════════════════════════════════

def _method_sampler(dit, scheduler, method: str, head: CacheHead, config: CacheHeadConfig,
                    cfg_scale: float, patch_size, grid):
    """Build the sampler for a method (full / carry / trained arm)."""
    if method == METHOD_FULL:
        n = config.schedule.num_inference_steps
        schedule = CacheHeadSchedule(num_inference_steps=n, full_step_indices=tuple(range(1, n + 1)))
        return HybridSampler(dit, scheduler, head, schedule, cfg_scale, patch_size, grid)
    if method == METHOD_CARRY:
        zero = CacheHead(CacheHeadConfig(model_id=config.model_id, cfg_scale=cfg_scale))
        return HybridSampler(dit, scheduler, zero, config.schedule, cfg_scale, patch_size, grid)
    return HybridSampler(dit, scheduler, head, config.schedule, cfg_scale, patch_size, grid)


def capture_panel(
    pipe,
    head: CacheHead,
    config: CacheHeadConfig,
    prompts: list[str],
    seeds: list[int],
    methods: list[str],
    cfg_scale: float,
    run_cache_dir: str | Path,
    fingerprint: str,
    device: str,
    dtype: torch.dtype,
) -> tuple[dict[str, np.ndarray], dict]:
    """Run the panel; returns (states_by_method, metadata).

    ``states_by_method[m]`` is [P, steps, C, F, H, W] with runs ordered
    prompt-major then seed.  Caches per (method, prompt, seed) and resumes.
    """
    from diffsynth.models.wan_video_dit import set_to_torch_norm

    dit = pipe.dit
    scheduler = pipe.scheduler
    set_to_torch_norm([dit])
    dit.eval()
    dit.requires_grad_(False)

    z_dim = pipe.vae.model.z_dim
    num_frames = 81
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = 60  # 480 // 8
    latent_w = 104  # 832 // 8
    shape = (1, z_dim, latent_frames, latent_h, latent_w)
    grid = (latent_frames, latent_h // dit.patch_size[1], latent_w // dit.patch_size[2])

    scheduler.set_timesteps(config.schedule.num_inference_steps, denoising_strength=1.0, shift=5.0)
    run_cache_dir = Path(run_cache_dir)
    run_cache_dir.mkdir(parents=True, exist_ok=True)

    def encode(text):
        ids, mask = pipe.tokenizer(text, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = pipe.text_encoder(ids, mask)
        for v in seq_lens:
            emb[:, v:] = 0
        return emb

    negative_prompt = (
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )
    ctx_nega = encode(negative_prompt)

    states_by_method: dict[str, np.ndarray] = {}
    metadata: dict = {"prompt_ids": [], "seeds": []}
    n_steps = config.schedule.num_inference_steps + 1

    with torch.inference_mode():
        for method in methods:
            sampler = _method_sampler(dit, scheduler, method, head, config, cfg_scale,
                                      dit.patch_size, grid)
            all_states = []
            for pi, prompt in enumerate(prompts):
                ctx_posi = encode(prompt)
                for si, seed in enumerate(seeds):
                    cache_path = run_cache_dir / f"{method}_p{pi:02d}_s{seed:03d}.npz"
                    cached = _load_cached_run(
                        cache_path, seed=seed, fingerprint=fingerprint,
                        expected_steps=n_steps, expected_latent_shape=shape[1:],
                    )
                    if cached is not None:
                        traj = cached[0]
                    else:
                        latents = pipe.generate_noise(shape, seed=seed, rand_device="cpu")
                        latents = latents.to(dtype=dtype, device=device)
                        _, traj_states, _ = sampler.sample(latents, ctx_posi, ctx_nega)
                        traj = np.stack([s.squeeze(0).numpy() for s in traj_states])
                        _save_run_cache(
                            cache_path, seed=seed, fingerprint=fingerprint,
                            trajectory=traj,
                            final_latents=traj[-1],
                        )
                    all_states.append(traj)
            states_by_method[method] = np.stack(all_states, axis=0)
            metadata["prompt_ids"] = [str(pi) for pi in range(len(prompts))]
            metadata["seeds"] = list(seeds)

    timesteps = scheduler.timesteps.detach().cpu().numpy()
    sigmas = scheduler.sigmas.detach().cpu().numpy()
    metadata["timesteps"] = timesteps
    metadata["sigmas"] = sigmas
    return states_by_method, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="CacheHead PCA trajectory-difference evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompts-jsonl", required=True, help="held-out prompts (id, caption)")
    parser.add_argument("--panel-size", type=int, default=8)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cfg", type=float, default=None, help="override checkpoint CFG scale")
    parser.add_argument("--method-name", default=None, help="label for the evaluated arm")
    parser.add_argument("--out-dir", default="pca_output")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    head, config = load_cache_head(args.checkpoint, device=device)
    head = head.to(dtype=dtype).eval()
    cfg_scale = args.cfg if args.cfg is not None else config.cfg_scale
    arm_name = args.method_name or Path(args.checkpoint).stem

    prompts = []
    with open(args.prompts_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                prompts.append(json.loads(line)["caption"])
    prompts = prompts[: args.panel_size]
    seeds = [int(s) for s in args.seeds.split(",")]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id=config.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=config.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=config.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id=config.model_id, origin_file_pattern="google/umt5-xxl/"),
    )

    methods = [METHOD_FULL, METHOD_CARRY, arm_name]
    fingerprint = _cache_fingerprint()  # settings that invalidate cached runs
    states, metadata = capture_panel(
        pipe, head, config, prompts, seeds, methods, cfg_scale,
        run_cache_dir=Path(args.out_dir) / "runs", fingerprint=fingerprint,
        device=device, dtype=dtype,
    )

    fit, coords = build_shared_pca(states, pca_seed=0)
    metrics = trajectory_metrics(states, coords)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = config.schedule.num_inference_steps
    full_anchors = [i for i in range(n + 1) if (i) in config.schedule.full_step_indices]
    head_steps = [i for i in range(n) if not config.schedule.is_full_step(i)]
    make_trajectory_plot(coords, fit["explained_variance_ratio"], full_anchors, head_steps,
                         str(out_dir / "pca_trajectory.png"))
    save_artifacts(
        str(out_dir / "pca_trajectory.npz"),
        str(out_dir / "pca_trajectory.png"),
        str(out_dir / "trajectory_metrics.json"),
        fit=fit, per_method_coords=coords, method_labels=methods,
        prompt_ids=metadata["prompt_ids"], seeds=metadata["seeds"],
        timesteps=metadata["timesteps"], sigmas=metadata["sigmas"], metrics=metrics,
    )
    print(f"Wrote {out_dir}/pca_trajectory.npz, .png, trajectory_metrics.json")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
