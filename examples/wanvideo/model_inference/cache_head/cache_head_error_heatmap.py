"""Per-patch CacheHead-vs-teacher error heat maps over a full trajectory.

Rolls one hybrid trajectory and, at *every* denoising step, compares the head's
prediction against what frozen Wan would have produced at that same state:

    err[k, b, s] = || v_head[k, b, s, :] - v_teacher[k, b, s, :] ||_2

over the 64 noise-token channels, reshaped from the token index ``s`` back to
the ``(f, h, w)`` token grid.  The result answers "where, and when, does the
head diverge from the teacher" -- the spatial complement to the per-head-step
validation loss, which only says *when*.

The head prediction is recorded at full steps too, where the head is not used
to advance the trajectory.  Those are counterfactuals ("what would the head
have said here"), and they are exactly the steps a schedule change might hand
to the head next, so they are worth seeing.

The trajectory itself always advances the real hybrid way: teacher at anchor
steps, head at head steps.

``collect_step_errors`` is pure tensor work and is CPU-testable with a fake dit
and scheduler.  ``render_error_heatmap`` imports matplotlib lazily so this
module stays importable without it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cache_head_model import CacheHeadSchedule, load_cache_head
from cache_head_model_inference import full_step, head_step


# ═══════════════════════════════════════════════════════════════
# Error collection
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_step_errors(
    *,
    dit,
    scheduler,
    head,
    schedule: CacheHeadSchedule,
    cfg_scale: float,
    patch_size,
    grid: tuple[int, int, int],
    latents: torch.Tensor,
    ctx: torch.Tensor,
    neg_ctx: torch.Tensor,
) -> dict:
    """Roll one hybrid trajectory, recording per-patch head-vs-teacher error.

    Returns a dict with
        ``errors``        [num_steps, B, f, h, w] absolute L2 error per patch
        ``rel_errors``    same, divided by the teacher's own token norm
        ``is_head``       [num_steps] bool, True where the head drives the step
        ``has_pred``      [num_steps] bool, False before the first full step
        ``timesteps``     [num_steps] the Wan timestep at each step
        ``final_latents`` [B, C, F, H, W] the rollout's output, ready to decode
    """
    f, h, w = grid
    n_batch = latents.shape[0]
    num_steps = schedule.num_inference_steps
    if neg_ctx.shape[0] != n_batch:
        neg_ctx = neg_ctx.expand(n_batch, *neg_ctx.shape[1:])

    errors = torch.full((num_steps, n_batch, f, h, w), float("nan"))
    rel_errors = torch.full((num_steps, n_batch, f, h, w), float("nan"))
    is_head = torch.zeros(num_steps, dtype=torch.bool)
    has_pred = torch.zeros(num_steps, dtype=torch.bool)
    timesteps = torch.zeros(num_steps)

    prev_guided = None
    for k in range(num_steps):
        t = scheduler.timesteps[k].reshape(1).to(device=latents.device, dtype=latents.dtype)
        timesteps[k] = float(scheduler.timesteps[k])
        is_head[k] = schedule.is_head_step(k)

        # Teacher's answer at this exact state, needed at every step.
        teacher_noise, teacher_tokens = full_step(dit, latents, t, ctx, neg_ctx, cfg_scale)

        if prev_guided is not None:
            head_noise, head_tokens = head_step(
                head, t, prev_guided, grid, patch_size,
                current_latents=latents, context=ctx,
            )
            diff = (head_tokens - teacher_tokens).float()          # [B, S, C]
            err = diff.norm(dim=-1)                                # [B, S]
            rel = err / teacher_tokens.float().norm(dim=-1).clamp_min(1e-6)
            errors[k] = err.reshape(n_batch, f, h, w).cpu()
            rel_errors[k] = rel.reshape(n_batch, f, h, w).cpu()
            has_pred[k] = True
        else:
            head_tokens = None

        # Advance the true hybrid trajectory.
        if schedule.is_full_step(k):
            noise_pred, prev_guided = teacher_noise, teacher_tokens
        else:
            if head_tokens is None:
                raise RuntimeError(
                    f"head step at progress {k} before any full step; invalid schedule {schedule}"
                )
            noise_pred, prev_guided = head_noise, head_tokens
        latents = scheduler.step(noise_pred, t, latents)

    return {
        "errors": errors,
        "rel_errors": rel_errors,
        "is_head": is_head,
        "has_pred": has_pred,
        "timesteps": timesteps,
        # The heat map says where the head diverges; the video says whether it
        # matters.  Same rollout, so the two always describe the same run.
        "final_latents": latents,
    }


def step_summary(result: dict, relative: bool = False) -> dict:
    """Per-step mean and p90 error across batch and patches (NaN-safe)."""
    data = result["rel_errors"] if relative else result["errors"]
    means, p90s = [], []
    for k in range(data.shape[0]):
        flat = data[k].flatten()
        flat = flat[~torch.isnan(flat)]
        if flat.numel() == 0:
            means.append(float("nan"))
            p90s.append(float("nan"))
        else:
            means.append(float(flat.mean()))
            p90s.append(float(flat.quantile(0.9)))
    return {"mean": means, "p90": p90s}


# ═══════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════

# Sequential magnitude data gets ONE perceptually-uniform ramp, monotonic in
# lightness, so a darker patch always means a larger error.  Explicitly not a
# rainbow map (`jet` and friends invent edges where the data has none).
_SEQUENTIAL_CMAP = "magma"


def print_prompts(prompts: list[str], *, prefix: str = "[heatmap]") -> None:
    """Print the exact captions represented by a heat-map batch."""
    print(f"{prefix} prompts ({len(prompts)}):")
    for index, prompt in enumerate(prompts, start=1):
        print(f"  [{index}] {prompt}")


def render_error_heatmap(
    result: dict,
    out_path: str | Path,
    *,
    relative: bool = False,
    title: str | None = None,
    frame: int | None = None,
) -> Path:
    """Render the per-step panel grid plus a per-step summary curve.

    All panels share one color scale, so a patch that looks hotter at step 12
    than at step 3 *is* hotter; per-panel normalization would destroy exactly
    the comparison this figure exists to make.  ``frame`` selects one latent
    frame; the default averages over them.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = result["rel_errors"] if relative else result["errors"]
    num_steps = data.shape[0]
    # Average over the batch; select or average over latent frames.
    panels = data.mean(dim=1)                      # [steps, f, h, w]
    panels = panels[:, frame] if frame is not None else panels.mean(dim=1)   # [steps, h, w]

    finite = panels[~torch.isnan(panels)]
    vmax = float(finite.quantile(0.99)) if finite.numel() else 1.0
    vmax = vmax if vmax > 0 else 1.0

    n_cols = 5
    n_rows = (num_steps + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(2.6 * n_cols + 1.2, 2.5 * n_rows + 3.0))
    gs = GridSpec(n_rows + 1, n_cols + 1, figure=fig,
                  width_ratios=[1] * n_cols + [0.06],
                  height_ratios=[1] * n_rows + [0.9], hspace=0.35, wspace=0.12)

    image = None
    for k in range(num_steps):
        ax = fig.add_subplot(gs[k // n_cols, k % n_cols])
        panel = panels[k]
        if torch.isnan(panel).all():
            ax.text(0.5, 0.5, "no head\nprediction yet", ha="center", va="center",
                    fontsize=8, color="#888888", transform=ax.transAxes)
            ax.set_facecolor("#f2f2f2")
        else:
            image = ax.imshow(panel.numpy(), cmap=_SEQUENTIAL_CMAP, vmin=0.0, vmax=vmax,
                              interpolation="nearest", aspect="auto")
        driver = "head" if bool(result["is_head"][k]) else "FULL"
        # Identity is never color-alone: each panel says which model drove it.
        ax.set_title(f"step {k + 1}  ·  {driver}\nt={float(result['timesteps'][k]):.0f}",
                     fontsize=8,
                     color="#222222" if driver == "head" else "#0b6fa4")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#0b6fa4" if driver == "FULL" else "#dddddd")
            spine.set_linewidth(2.0 if driver == "FULL" else 0.8)

    if image is not None:
        cax = fig.add_subplot(gs[:n_rows, n_cols])
        cbar = fig.colorbar(image, cax=cax)
        cbar.set_label(("relative" if relative else "absolute") + " per-patch L2 error",
                       fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Per-step summary: the "when" that pairs with the panels' "where".
    summary = step_summary(result, relative=relative)
    ax = fig.add_subplot(gs[n_rows, :n_cols])
    steps = list(range(1, num_steps + 1))
    ax.plot(steps, summary["mean"], linewidth=2.0, color="#7b2d8e", label="mean")
    ax.plot(steps, summary["p90"], linewidth=2.0, color="#c46a2f",
            linestyle="--", label="p90")
    for k in range(num_steps):
        if not bool(result["is_head"][k]):
            ax.axvspan(k + 0.5, k + 1.5, color="#0b6fa4", alpha=0.08, linewidth=0)
    ax.set_xlabel("denoising step (1-indexed; shaded = full Wan anchor)", fontsize=9)
    ax.set_ylabel(("relative " if relative else "") + "error", fontsize=9)
    ax.set_xticks(steps)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=8, frameon=False)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_error_arrays(result: dict, out_path: str | Path) -> Path:
    """Persist the raw error tensors so panels can be re-rendered without Wan.

    ``final_latents`` is deliberately excluded: it is large, lives on the
    accelerator, and is already represented by the decoded video.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {k: v for k, v in result.items() if k != "final_latents"},
        out_path,
    )
    return out_path


def save_rollout_video(
    pipe, latents: torch.Tensor, out_path: str | Path, *, device, fps: int = 15,
    quality: int = 5, tile_size=(30, 52), tile_stride=(15, 26),
) -> list[Path]:
    """Decode the rollout's latents and write one mp4 per prompt.

    Mirrors the decode in ``cache_head_model_inference.run_pipeline`` so the
    video beside a heat map is the same artifact the inference runner produces.
    """
    from diffsynth.utils.data import save_video

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    n_batch = latents.shape[0]
    for i in range(n_batch):
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
        video = pipe.vae.decode(
            latents[i : i + 1].to(device=device),
            device=device,
            tiled=True,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        # One prompt keeps the requested name; a batch gets an index suffix.
        target = out_path if n_batch == 1 else out_path.with_name(
            f"{out_path.stem}-{i}{out_path.suffix}"
        )
        save_video(pipe.vae_output_to_video(video), str(target), fps=fps, quality=quality)
        written.append(target)
    return written


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render per-patch CacheHead-vs-teacher error heat maps",
    )
    parser.add_argument("--checkpoint", required=True, help="CacheHead checkpoint")
    parser.add_argument("--captions", required=True, help="caption JSONL")
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--relative", action="store_true",
                        help="normalize by the teacher's token norm")
    parser.add_argument("--frame", type=int, default=None,
                        help="latent frame to show (default: average over frames)")
    parser.add_argument("--out-dir", default="cache_head_heatmaps")
    parser.add_argument("--video", default=None,
                        help="path for the decoded rollout video "
                             "(default: <out-dir>/rollout.mp4; one file per prompt, "
                             "index-suffixed when --num-prompts > 1)")
    parser.add_argument("--no-video", action="store_true",
                        help="skip VAE decoding and write only the heat map")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--no-network", action="store_true",
                        help="offline mode: never contact modelscope/HuggingFace. "
                             "Model files must already sit under "
                             "--model-base-path/<model-id>/")
    parser.add_argument("--model-base-path", default=None,
                        help="local model root for --no-network (default: "
                             "DIFFSYNTH_MODEL_BASE_PATH, else ./models)")
    args = parser.parse_args()

    # Must precede the diffsynth/transformers imports below.
    from cache_head_model_training import apply_no_network
    apply_no_network(args)

    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.models.wan_video_dit import set_to_torch_norm
    from cache_head_model_training import PromptDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    torch.manual_seed(args.seed)

    model_kwargs = {"skip_download": True} if args.no_network else {}
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors", **model_kwargs),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **model_kwargs),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth", **model_kwargs),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="google/umt5-xxl/", **model_kwargs),
    )
    dit = pipe.dit
    set_to_torch_norm([dit])
    dit.eval()
    dit.requires_grad_(False)

    head, config = load_cache_head(args.checkpoint, device=device)
    head = head.to(device=device, dtype=dtype).eval()
    schedule = config.schedule

    @torch.no_grad()
    def encode(text):
        ids, mask = pipe.tokenizer(text, return_mask=True, add_special_tokens=True)
        ids, mask = ids.to(pipe.device), mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            emb[i, v:] = 0
        return emb.detach()

    dataset = PromptDataset(args.captions, split=args.split)
    captions = [caption for _, caption in dataset.items[:args.num_prompts]]
    if not captions:
        raise ValueError(f"no captions found in split {args.split!r}")
    print_prompts(captions)
    ctx = encode(captions)
    neg_ctx = encode(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    scheduler = pipe.scheduler
    scheduler.set_timesteps(schedule.num_inference_steps, denoising_strength=1.0, shift=5.0)

    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_h = args.height // pipe.vae.upsampling_factor
    latent_w = args.width // pipe.vae.upsampling_factor
    grid = (latent_frames, latent_h // dit.patch_size[1], latent_w // dit.patch_size[2])
    latents = torch.randn(
        len(captions), pipe.vae.model.z_dim, latent_frames, latent_h, latent_w,
        device=device, dtype=dtype,
    )

    result = collect_step_errors(
        dit=dit, scheduler=scheduler, head=head, schedule=schedule, cfg_scale=args.cfg,
        patch_size=dit.patch_size, grid=grid, latents=latents, ctx=ctx, neg_ctx=neg_ctx,
    )

    out_dir = Path(args.out_dir)
    png = render_error_heatmap(
        result, out_dir / "cache_head_error_heatmap.png",
        relative=args.relative, frame=args.frame,
        title=f"CacheHead vs frozen Wan · {len(captions)} prompts · seed {args.seed}",
    )
    pt = save_error_arrays(result, out_dir / "cache_head_error_heatmap.pt")
    print(f"wrote {png}")
    print(f"wrote {pt}")

    if not args.no_video:
        video_path = Path(args.video) if args.video else out_dir / "rollout.mp4"
        for i, written in enumerate(save_rollout_video(
            pipe, result["final_latents"], video_path, device=device, fps=args.fps,
        )):
            print(f"wrote {written}  <- {captions[i]!r}")


if __name__ == "__main__":
    main()
