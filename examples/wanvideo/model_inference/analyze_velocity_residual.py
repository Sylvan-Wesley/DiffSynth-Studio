"""
Velocity x Residual heuristic analysis for Wan2.1-T2V-1.3B.

Runs the *vanilla* Wan2.1 model (no NaviCache / RAS / MotionCache) for a fixed
number of denoising steps and, at each step, records the guided denoising
velocity ``v = noise_pred`` (CFG-combined, latent space ``[B, C, F, H, W]``).

For each consecutive pair ``(v_pre, v_cur)`` with residual ``r = v_cur - v_pre``
we measure three relationships between the cached previous velocity and the
residual:

1. Correlation:  cosine(v_pre, r), Pearson(v_pre, r), ||r|| / ||v_pre||.
2. Per-frame matrix rank of v_pre, r, v_cur  (effective channel dimensionality
   of each latent frame's velocity slice), via SVD.
3. Projection of r onto the subspace of v_pre: the 1-D span{v_pre} and the
   growing span{v_0..v_pre} of all prior velocities (incremental Gram-Schmidt).
4. EDM-style straightness: the per-step deviation of the instantaneous
   velocity from the trajectory's overall initial-to-final displacement.

Output: a per-step table, a ``velocity_analysis.npz`` with all scalar + per-frame
quantities, and (optionally) a matplotlib figure vs sigma with a marker at 0.9.

Usage:
    python analyze_velocity_residual.py

The model loads from Hugging Face. A full run needs a CUDA device (the script
auto-detects cuda -> mps -> cpu, but 50 steps at 480x832x81f is only practical on
CUDA). The pure analysis functions at module scope are importable and CPU-testable
without loading the model.
"""

import numpy as np
import torch
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

model_id = "Wan-AI/Wan2.1-T2V-1.3B"

num_inference_steps = 50
cfg_scale = 5.0
seed = 0
num_frames = 81
height = 480
width = 832

prompt = "纪实摄影风格画面，一只活泼的小狗在绿茵茵的草地上迅速奔跑。小狗毛色棕黄，两只耳朵立起，神情专注而欢快。阳光洒在它身上，使得毛发看上去格外柔软而闪亮。背景是一片开阔的草地，偶尔点缀着几朵野花，远处隐约可见蓝天和几片白云。透视感鲜明，捕捉小狗奔跑时的动感和四周草地的生机。中景侧面移动视角。"
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

out_npz = "velocity_analysis.npz"
out_png = "velocity_analysis.png"
plot = True

# Marked sigma for the "sharp turn" intuition (flow-matching sigma, 1 -> 0).
sigma_marker = 0.9


# ═══════════════════════════════════════════════════════════════
# Analysis primitives (pure, CPU, importable / testable)
# ═══════════════════════════════════════════════════════════════

def _flatten(v) -> np.ndarray:
    """Return a contiguous float32 1-D view of a velocity array [C, F, H, W]."""
    return np.ascontiguousarray(v, dtype=np.float32).reshape(-1)


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_flatten(a), _flatten(b)))


def _norm(v) -> float:
    return float(np.linalg.norm(_flatten(v)))


def cosine(a, b) -> float:
    """Cosine similarity of two flattened vectors."""
    denom = _norm(a) * _norm(b)
    if denom == 0.0:
        return 0.0
    return _dot(a, b) / denom


def pearson(a, b) -> float:
    """Pearson correlation of two flattened vectors (mean-centered cosine)."""
    a = _flatten(a); b = _flatten(b)
    a = a - a.mean(); b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def relative_norm(r, v_pre) -> float:
    """||r|| / ||v_pre||."""
    return float(_norm(r) / (_norm(v_pre) + 1e-12))


def per_frame_rank(v):
    """Per-frame numerical (matrix) rank of a velocity array [C, F, H, W].

    For each latent frame f, forms the frame matrix M = v[:, f].reshape(C, H*W)
    and computes its SVD. Returns:
        ranks  [F] int     -- number of singular values above the numpy
                              matrix_rank tolerance (s_max * max(shape) * eps)
        stable [F] float   -- stable rank (sum(s))^2 / sum(s^2)
        svals  [F, C] float-- sorted singular values per frame
    """
    v = np.ascontiguousarray(v, dtype=np.float32)
    C, F, H, W = v.shape
    ranks = np.zeros(F, dtype=np.int64)
    stable = np.zeros(F, dtype=np.float64)
    svals = np.zeros((F, C), dtype=np.float64)
    for f in range(F):
        M = v[:, f, :, :].reshape(C, -1)
        s = np.linalg.svd(M, compute_uv=False)
        svals[f, : s.shape[0]] = s
        if s.shape[0] > 0 and s[0] > 0:
            tol = s[0] * max(M.shape) * np.finfo(np.float32).eps
            ranks[f] = int((s > tol).sum())
        stable[f] = float(s.sum() ** 2 / (np.sum(s ** 2) + 1e-30))
    return ranks, stable, svals


def project_fraction_1d(r, v_pre) -> float:
    """Fraction of ||r||^2 lying along span{v_pre} (== cos^2(r, v_pre))."""
    nr2 = _dot(r, r)
    if nr2 == 0.0:
        return 0.0
    nv2 = _dot(v_pre, v_pre)
    if nv2 == 0.0:
        return 0.0
    return float((_dot(r, v_pre) ** 2) / (nr2 * nv2))


def edm_straightness_metrics(velocities, initial_latents, final_latents, sigma_start):
    """Return EDM-style curvature and chord cosine for every denoising step.

    Flow matching integrates ``dx / d sigma = v`` from ``sigma_start`` down to
    zero.  In the straight-trajectory case, ``sigma_start * v_i`` equals the
    global chord ``x_initial - x_final`` at every step.  The returned
    curvature is the EDM-toy mean squared deviation from that chord, and the
    cosine records their directional agreement.
    """
    if len(velocities) == 0:
        raise ValueError("velocities must contain at least one denoising step.")
    if sigma_start <= 0:
        raise ValueError(f"sigma_start must be positive, got {sigma_start}.")

    chord = np.asarray(initial_latents, dtype=np.float32) - np.asarray(final_latents, dtype=np.float32)
    chord_flat = _flatten(chord)
    curvature = np.empty(len(velocities), dtype=np.float64)
    cosine_to_chord = np.empty(len(velocities), dtype=np.float64)
    for i, velocity in enumerate(velocities):
        instantaneous = float(sigma_start) * _flatten(velocity)
        difference = instantaneous - chord_flat
        curvature[i] = float(np.dot(difference, difference) / difference.size)
        cosine_to_chord[i] = cosine(instantaneous, chord_flat)
    return curvature, cosine_to_chord


class GrowingSpan:
    """Incremental orthonormal basis of an accumulating set of velocity vectors.

    ``Q`` holds the orthonormal rows of span{v_0..v_k}. ``project_fraction``
    returns how much of a residual's energy lies in the current span, and ``add``
    folds the next velocity into the span (modified Gram-Schmidt, one
    re-orthogonalization pass).
    """

    def __init__(self):
        self.Q = None  # [k, N] float32, orthonormal rows

    def project_fraction(self, r) -> float:
        r = _flatten(r)
        nr2 = float(np.dot(r, r))
        if nr2 == 0.0 or self.Q is None:
            return 0.0
        c = self.Q @ r            # [k]
        proj = self.Q.T @ c       # [N]
        return float(np.dot(proj, proj) / nr2)

    def add(self, v) -> None:
        v = _flatten(v)
        if self.Q is None:
            nv = np.linalg.norm(v)
            if nv > 0:
                self.Q = (v / nv)[None, :]
            return
        # Subtract projection onto current span, then re-orthogonalize once.
        u = v - self.Q.T @ (self.Q @ v)
        u = u - self.Q.T @ (self.Q @ u)
        nu = np.linalg.norm(u)
        if nu > 1e-10:
            self.Q = np.vstack([self.Q, (u / nu)[None, :]])


# ═══════════════════════════════════════════════════════════════
# Prompt encoding + model load
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_prompt(pipe, prompt: str) -> torch.Tensor:
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    # Keep the analysis helpers importable for CPU tests without requiring the
    # complete DiffSynth inference dependency stack.
    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.models.wan_video_dit import set_to_torch_norm

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

    dit = pipe.dit
    scheduler = pipe.scheduler

    set_to_torch_norm([dit])
    print(f"  DiT blocks: {len(dit.blocks)}, dim: {dit.dim}, patch: {dit.patch_size}")

    print("Encoding prompts...")
    ctx_posi = encode_prompt(pipe, prompt)
    ctx_nega = encode_prompt(pipe, negative_prompt)

    print("Initializing latents...")
    z_dim = pipe.vae.model.z_dim
    latent_frames = (num_frames - 1) // 4 + 1
    latent_h = height // pipe.vae.upsampling_factor
    latent_w = width // pipe.vae.upsampling_factor
    shape = (1, z_dim, latent_frames, latent_h, latent_w)
    latents = pipe.generate_noise(shape, seed=seed, rand_device="cpu")
    latents = latents.to(dtype=pipe.torch_dtype, device=device)
    initial_latents = latents.detach().float().cpu().squeeze(0).numpy()

    scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)
    sigmas = scheduler.sigmas.detach().cpu().numpy()
    timesteps = scheduler.timesteps.detach().cpu().numpy()
    print(f"  Timesteps: {len(scheduler.timesteps)} steps, "
          f"sigma range [{sigmas[0]:.3f}, {sigmas[-1]:.3f}]")

    dit.eval()

    # Record the velocity at every step (CPU float32) for post-hoc analysis.
    velocities = []  # list of [C, F, H, W] float32 numpy arrays

    print(f"\nDenoising ({num_inference_steps} steps, CFG {cfg_scale})...")
    with torch.inference_mode():
        for progress_id, timestep in enumerate(tqdm(scheduler.timesteps)):
            t = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=device)

            noise_posi = dit(x=latents, timestep=t, context=ctx_posi)
            noise_nega = dit(x=latents, timestep=t, context=ctx_nega)
            noise_pred = noise_nega + cfg_scale * (noise_posi - noise_nega)

            velocities.append(noise_pred.detach().float().cpu().squeeze(0).numpy())

            latents = scheduler.step(
                noise_pred,
                scheduler.timesteps[progress_id],
                latents,
            )

    final_latents = latents.detach().float().cpu().squeeze(0).numpy()
    edm_curvature, cos_instant_straight = edm_straightness_metrics(
        velocities, initial_latents, final_latents, sigma_start=float(sigmas[0]),
    )

    # ═══════════════════════════════════════════════════════════
    # Analysis over consecutive pairs
    # ═══════════════════════════════════════════════════════════

    n_pairs = len(velocities) - 1
    F = velocities[0].shape[1]   # latent temporal frames
    C = velocities[0].shape[0]   # latent channels

    pair_sigmas = sigmas[1:]                       # sigma of v_cur for each pair
    pair_timesteps = timesteps[1:]

    cos_pre_res = np.zeros(n_pairs)
    cos_pre_cur = np.zeros(n_pairs)
    pear_pre_res = np.zeros(n_pairs)
    rel_norm = np.zeros(n_pairs)
    f_par_1d = np.zeros(n_pairs)
    f_par_grow = np.zeros(n_pairs)

    ranks = np.zeros((n_pairs, F, 3), dtype=np.int64)        # pre, res, cur
    stable = np.zeros((n_pairs, F, 3), dtype=np.float64)
    svals = np.zeros((n_pairs, F, 3, C), dtype=np.float64)

    span = GrowingSpan()
    span.add(velocities[0])                      # span{v_0} for the first pair

    for i in range(n_pairs):
        v_pre = velocities[i]
        v_cur = velocities[i + 1]
        r = v_cur - v_pre

        cos_pre_res[i] = cosine(v_pre, r)
        cos_pre_cur[i] = cosine(v_pre, v_cur)
        pear_pre_res[i] = pearson(v_pre, r)
        rel_norm[i] = relative_norm(r, v_pre)
        f_par_1d[i] = project_fraction_1d(r, v_pre)
        f_par_grow[i] = span.project_fraction(r)

        ranks[i, :, 0], stable[i, :, 0], svals[i, :, 0, :] = per_frame_rank(v_pre)
        ranks[i, :, 1], stable[i, :, 1], svals[i, :, 1, :] = per_frame_rank(r)
        ranks[i, :, 2], stable[i, :, 2], svals[i, :, 2, :] = per_frame_rank(v_cur)

        span.add(v_cur)

    # ═══════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════

    header = (
        f"{'step':>4} {'sigma':>7} {'t':>5} "
        f"{'cos(pre,r)':>10} {'pearson':>8} {'||r||/||pre||':>12} "
        f"{'rk_pre':>6} {'rk_res':>6} {'rk_cur':>6} "
        f"{'f_par_1d':>9} {'f_par_grow':>10}"
    )
    print("\n" + header)
    print("-" * len(header))
    for i in range(n_pairs):
        print(
            f"{i:>4} {pair_sigmas[i]:>7.3f} {int(pair_timesteps[i]):>5} "
            f"{cos_pre_res[i]:>10.4f} {pear_pre_res[i]:>8.4f} {rel_norm[i]:>12.4f} "
            f"{ranks[i, :, 0].mean():>6.1f} {ranks[i, :, 1].mean():>6.1f} {ranks[i, :, 2].mean():>6.1f} "
            f"{f_par_1d[i]:>9.4f} {f_par_grow[i]:>10.4f}"
        )

    print("\nEDM-style instantaneous velocity vs. overall straight-line displacement")
    print(f"{'step':>4} {'sigma':>7} {'t':>5} {'curvature':>12} {'cos(v,chord)':>13}")
    print("-" * 49)
    for i in range(len(velocities)):
        print(
            f"{i:>4} {sigmas[i]:>7.3f} {int(timesteps[i]):>5} "
            f"{edm_curvature[i]:>12.6g} {cos_instant_straight[i]:>13.4f}"
        )

    np.savez(
        out_npz,
        sigmas=sigmas,
        timesteps=timesteps,
        pair_sigmas=pair_sigmas,
        pair_timesteps=pair_timesteps,
        cos_pre_res=cos_pre_res,
        cos_pre_cur=cos_pre_cur,
        pear_pre_res=pear_pre_res,
        rel_norm=rel_norm,
        f_par_1d=f_par_1d,
        f_par_grow=f_par_grow,
        edm_curvature=edm_curvature,
        cos_instant_straight=cos_instant_straight,
        ranks=ranks,
        stable_ranks=stable,
        singular_values=svals,
    )
    print(f"\nSaved analysis to: {out_npz}")

    if plot:
        _make_plot(pair_sigmas, cos_pre_res, pear_pre_res, rel_norm,
                   f_par_1d, f_par_grow, ranks, sigmas, edm_curvature,
                   cos_instant_straight, out_png, sigma_marker)
        print(f"Saved plot to: {out_png}")


def _make_plot(sigmas, cos_pre_res, pear_pre_res, rel_norm,
               f_par_1d, f_par_grow, ranks, step_sigmas, edm_curvature,
               cos_instant_straight, out_png, marker):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(21, 8), constrained_layout=True)
    x = sigmas  # descending 1 -> 0

    def _plot(ax, y, title, x_values=None):
        x_values = x if x_values is None else x_values
        ax.plot(x_values, y, "o-", markersize=3)
        ax.axvline(marker, color="r", linestyle="--", alpha=0.6, label=f"sigma={marker}")
        ax.set_xlabel("sigma"); ax.set_title(title); ax.grid(alpha=0.3)
        ax.invert_xaxis()
        ax.legend()

    _plot(axes[0, 0], cos_pre_res, "cos(v_pre, residual)")
    _plot(axes[0, 1], pear_pre_res, "Pearson(v_pre, residual)")
    _plot(axes[0, 2], rel_norm, "||residual|| / ||v_pre||")
    _plot(
        axes[0, 3], edm_curvature,
        "EDM curvature: ||sigma_0 v - chord||^2 / D",
        step_sigmas,
    )
    _plot(axes[1, 0], f_par_1d, "residual energy in span{v_pre}")
    _plot(axes[1, 1], f_par_grow, "residual energy in growing span")
    _plot(axes[1, 2], ranks[..., 1].mean(axis=1), "mean per-frame rank(residual)")
    # also overlay pre/cur rank means on the last panel
    axes[1, 2].plot(x, ranks[..., 0].mean(axis=1), "s--", markersize=3, label="v_pre")
    axes[1, 2].plot(x, ranks[..., 2].mean(axis=1), "^--", markersize=3, label="v_cur")
    axes[1, 2].legend()

    axes[1, 3].plot(step_sigmas, cos_instant_straight, "o-", markersize=3)
    axes[1, 3].axvline(marker, color="r", linestyle="--", alpha=0.6, label=f"sigma={marker}")
    axes[1, 3].set_xlabel("sigma")
    axes[1, 3].set_title("cos(instantaneous velocity, chord)")
    axes[1, 3].grid(alpha=0.3)
    axes[1, 3].invert_xaxis()
    axes[1, 3].legend()

    fig.savefig(out_png, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
