"""
Runs a seed sweep with the vanilla Wan2.1-T2V-1.3B sampler, records the
latent state immediately before each denoising model call, and embeds every
run-step state into one shared PCA plane.  Each run is then drawn as a line in
denoising order.

Outputs:
    latent_pca_trajectory.npz -- PCA coordinates and metadata (not raw latents)
    latent_pca_trajectory.png -- shared PC1/PC2 trajectory plot
    geometric_trajectory_runs/ -- resumable per-seed latents and MP4 videos

Usage:
    python analyze_geometric_trajectory.py

The pure PCA functions can be imported and CPU-tested without loading a model.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

model_id = "Wan-AI/Wan2.1-T2V-1.3B"

num_inference_steps = 50
cfg_scale = 5.0
seeds = tuple(range(10))
num_frames = 81
height = 480
width = 832

prompt = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

out_npz = "latent_pca_trajectory.npz"
out_png = "latent_pca_trajectory.png"
run_cache_dir = Path("geometric_trajectory_runs")
save_run_videos = True
pca_seed = 0


# ═══════════════════════════════════════════════════════════════
# Pure PCA helpers (CPU-testable)
# ═══════════════════════════════════════════════════════════════

def _validate_trajectories(trajectories: np.ndarray) -> np.ndarray:
    """Validate and normalize latent trajectories to float32 [run, step, ...]."""
    trajectories = np.asarray(trajectories, dtype=np.float32)
    if trajectories.ndim < 3:
        raise ValueError(
            "trajectories must have shape [runs, steps, feature...] with at least one feature dimension"
        )
    if trajectories.shape[0] < 1 or trajectories.shape[1] < 1:
        raise ValueError("trajectories must contain at least one run and one step")
    if not np.isfinite(trajectories).all():
        raise ValueError("trajectories must contain only finite values")
    return np.ascontiguousarray(trajectories)


def _pca_compute_device() -> torch.device:
    """Use CUDA for high-dimensional PCA when available; avoid MPS-only kernels."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fork_rng_devices(device: torch.device) -> list[int]:
    """Return explicit CUDA ordinals required by ``torch.random.fork_rng``."""
    if device.type != "cuda":
        return []
    return [torch.cuda.current_device() if device.index is None else device.index]


def _cache_fingerprint() -> str:
    """Identify settings whose changes make a saved denoising run stale."""
    values = (model_id, prompt, negative_prompt, num_inference_steps, cfg_scale, num_frames, height, width)
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()


def _load_cached_run(
    cache_path: Path,
    *,
    seed: int,
    fingerprint: str,
    expected_steps: int,
    expected_latent_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load one complete, matching seed run, or return ``None`` to regenerate it."""
    if not cache_path.is_file():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            if int(cache["seed"]) != seed or str(cache["fingerprint"]) != fingerprint:
                return None
            trajectory = np.asarray(cache["trajectory"], dtype=np.float32)
            final_latents = np.asarray(cache["final_latents"], dtype=np.float32)
    except (KeyError, OSError, ValueError):
        return None

    if trajectory.shape != (expected_steps,) + expected_latent_shape:
        return None
    if final_latents.shape != expected_latent_shape:
        return None
    if not np.isfinite(trajectory).all() or not np.isfinite(final_latents).all():
        return None
    return trajectory, final_latents


def _save_run_cache(
    cache_path: Path,
    *,
    seed: int,
    fingerprint: str,
    trajectory: np.ndarray,
    final_latents: np.ndarray,
) -> None:
    """Atomically save a completed run so an interrupted sweep can resume."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        seed=np.asarray(seed, dtype=np.int64),
        fingerprint=np.asarray(fingerprint),
        trajectory=np.asarray(trajectory, dtype=np.float32),
        final_latents=np.asarray(final_latents, dtype=np.float32),
    )
    temporary_path.replace(cache_path)


def fit_shared_pca_2d(
    trajectories: np.ndarray,
    *,
    random_seed: int = 0,
    device: torch.device | str | None = None,
) -> dict[str, np.ndarray]:
    """Fit one approximate 2-D PCA basis across all run-step latent states.

    ``torch.pca_lowrank`` avoids a prohibitively expensive full SVD for video
    latents while retaining a deterministic, shared basis for the plotted
    samples.  The return values are CPU NumPy arrays so callers can save and
    plot them without retaining a GPU allocation.
    """
    trajectories = _validate_trajectories(trajectories)
    runs, steps = trajectories.shape[:2]
    samples = trajectories.reshape(runs * steps, -1)
    if min(samples.shape) < 2:
        raise ValueError("PCA requires at least two samples and two features")

    pca_device = torch.device(device) if device is not None else _pca_compute_device()
    x = torch.from_numpy(samples).to(pca_device)
    mean = x.mean(dim=0)
    centered = x - mean

    # q > 2 lets the randomized solver estimate the leading two directions
    # robustly, while remaining small relative to the hundreds of thousands of
    # latent dimensions in a normal Wan run.
    q = min(8, centered.shape[0], centered.shape[1])
    with torch.random.fork_rng(devices=_fork_rng_devices(pca_device)):
        torch.manual_seed(random_seed)
        _, singular_values, right_vectors = torch.pca_lowrank(
            centered, q=q, center=False, niter=2
        )

    components = right_vectors[:, :2]
    coordinates = centered @ components

    # PCA axes have arbitrary signs.  Fixing each sign by its largest-magnitude
    # loading makes repeat plots comparable and enables deterministic tests.
    for component_id in range(2):
        pivot = torch.argmax(components[:, component_id].abs())
        if components[pivot, component_id] < 0:
            components[:, component_id].neg_()
            coordinates[:, component_id].neg_()

    denominator = max(centered.shape[0] - 1, 1)
    explained_variance = singular_values[:2].square() / denominator
    total_variance = centered.square().sum() / denominator
    explained_variance_ratio = explained_variance / total_variance

    return {
        "coordinates": coordinates.reshape(runs, steps, 2).cpu().numpy(),
        "components": components.cpu().numpy(),
        "mean": mean.cpu().numpy(),
        "explained_variance_ratio": explained_variance_ratio.cpu().numpy(),
    }


def _make_plot(
    coordinates: np.ndarray,
    seed_values: tuple[int, ...],
    explained_variance_ratio: np.ndarray,
    path: str,
) -> None:
    """Render seed trajectories as connected paths in the shared PCA plane."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    colors = plt.get_cmap("tab10", len(seed_values))
    for run_id, seed in enumerate(seed_values):
        x, y = coordinates[run_id, :, 0], coordinates[run_id, :, 1]
        color = colors(run_id)
        ax.plot(x, y, "-o", color=color, linewidth=1.5, markersize=3, label=f"seed {seed}")
        ax.scatter(x[0], y[0], s=64, marker="o", facecolors="none", edgecolors=color, zorder=3)
        ax.scatter(x[-1], y[-1], s=72, marker="X", color=color, zorder=3)

    ax.set_title("Wan latent denoising trajectories (start ○, final step ✕)")
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}% variance)")
    ax.grid(alpha=0.3)
    ax.legend(title="run", ncols=2)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════
# Wan sampling
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_prompt(pipe, text: str) -> torch.Tensor:
    ids, mask = pipe.tokenizer(text, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    embedding = pipe.text_encoder(ids, mask)
    for index, length in enumerate(seq_lens):
        embedding[:, length:] = 0
    return embedding


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    from diffsynth.models.wan_video_dit import set_to_torch_norm
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
    from diffsynth.utils.data import save_video

    device = _detect_device()
    print(f"Device: {device}")
    print("Loading model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id=model_id, origin_file_pattern="google/umt5-xxl/"),
    )
    set_to_torch_norm([pipe.dit])
    pipe.dit.eval()

    print("Encoding prompts...")
    ctx_posi = encode_prompt(pipe, prompt)
    ctx_nega = encode_prompt(pipe, negative_prompt)

    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = height // pipe.vae.upsampling_factor
    latent_w = width // pipe.vae.upsampling_factor
    shape = (1, pipe.vae.model.z_dim, latent_frames, latent_h, latent_w)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
    sigmas = scheduler.sigmas.detach().cpu().numpy()
    timesteps = scheduler.timesteps.detach().cpu().numpy()
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _cache_fingerprint()

    all_trajectories = None
    final_latents_by_run = []
    print(f"Capturing {len(seeds)} runs × {num_inference_steps} pre-step latent states...")
    with torch.inference_mode():
        for run_id, seed in enumerate(seeds):
            cache_path = run_cache_dir / f"seed_{seed:04d}.npz"
            cached_run = _load_cached_run(
                cache_path,
                seed=seed,
                fingerprint=fingerprint,
                expected_steps=num_inference_steps,
                expected_latent_shape=shape[1:],
            )
            if cached_run is None:
                latents = pipe.generate_noise(shape, seed=seed, rand_device="cpu")
                latents = latents.to(dtype=pipe.torch_dtype, device=device)
                run_states = []
                for progress_id, timestep in enumerate(tqdm(scheduler.timesteps, desc=f"seed {seed}")):
                    # This is the latent state aligned with timestep/sigma at progress_id.
                    run_states.append(latents.detach().float().cpu().squeeze(0).numpy())

                    t = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=device)
                    noise_posi = pipe.dit(x=latents, timestep=t, context=ctx_posi)
                    noise_nega = pipe.dit(x=latents, timestep=t, context=ctx_nega)
                    noise_pred = noise_nega + cfg_scale * (noise_posi - noise_nega)
                    latents = scheduler.step(noise_pred, scheduler.timesteps[progress_id], latents)

                run_trajectory = np.stack(run_states, axis=0)
                final_latents = latents.detach().float().cpu().squeeze(0).numpy()
                _save_run_cache(
                    cache_path,
                    seed=seed,
                    fingerprint=fingerprint,
                    trajectory=run_trajectory,
                    final_latents=final_latents,
                )
                print(f"Saved seed {seed} trajectory cache: {cache_path}")
            else:
                run_trajectory, final_latents = cached_run
                print(f"Reusing seed {seed} trajectory cache: {cache_path}")

            if all_trajectories is None:
                all_trajectories = np.empty((len(seeds),) + run_trajectory.shape, dtype=np.float32)
            all_trajectories[run_id] = run_trajectory
            final_latents_by_run.append(final_latents)

    if save_run_videos:
        print("Saving any missing per-seed videos...")
        pipe.dit.to("cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        with torch.inference_mode():
            for seed, final_latents in zip(seeds, final_latents_by_run):
                video_path = run_cache_dir / f"seed_{seed:04d}.mp4"
                if video_path.is_file():
                    print(f"Reusing video: {video_path}")
                    continue
                final_latents = torch.from_numpy(final_latents).unsqueeze(0).to(
                    dtype=pipe.torch_dtype, device=device
                )
                video = pipe.vae.decode(
                    final_latents,
                    device=device,
                    tiled=True,
                    tile_size=(30, 52),
                    tile_stride=(15, 26),
                )
                save_video(pipe.vae_output_to_video(video), video_path, fps=15, quality=5)
                print(f"Saved generated video: {video_path}")

    result = fit_shared_pca_2d(all_trajectories, random_seed=pca_seed)
    np.savez(
        out_npz,
        seeds=np.asarray(seeds, dtype=np.int64),
        timesteps=timesteps,
        sigmas=sigmas,
        coordinates=result["coordinates"],
        latent_shape=np.asarray(all_trajectories.shape[2:], dtype=np.int64),
        components=result["components"],
        mean=result["mean"],
        explained_variance_ratio=result["explained_variance_ratio"],
    )
    print(f"Saved PCA coordinates and metadata to: {out_npz}")

    _make_plot(result["coordinates"], seeds, result["explained_variance_ratio"], out_png)
    print(f"Saved trajectory plot to: {out_png}")


if __name__ == "__main__":
    main()
