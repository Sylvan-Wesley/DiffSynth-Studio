"""
Visualize RAS selection masks saved by RAS-Wan2.1-T2V-1.3B.py.

Reads the .npy mask files from ras_masks/ and produces:
  1. A grid image showing all steps' selection heatmaps.
  2. An animated GIF cycling through the heatmaps.

The mask directory must contain a single run (one consistent schedule);
a directory mixing files from multiple runs is refused (use --force to
override). Three mask layouts are understood, matching the RAS script's
viz_mask_mode:
  - frame_avg : mask_step_XX_t_YYY.npy            (one [h, w] map per step)
  - per_frame : mask_step_XX_f_KK_t_YYY.npy       (one [h, w] map per step+frame)
  - full_grid : mask_step_XX_f_all_t_YYY.npy      (one [f, h, w] grid per step)
For per_frame / full_grid, --frame N picks which video frame to show.

Usage:
    python visualize_ras_masks.py [--frame N]

Controls:
    --mask_dir     Directory containing mask_*.npy files (default: ras_masks)
    --output       Output image path (default: ras_masks_grid.png)
    --gif_output   Output GIF path (default: ras_masks.gif)
    --height       Video height in pixels (default: 480)
    --width        Video width in pixels (default: 832)
    --fps          GIF frames per second (default: 5)
    --cmap         Matplotlib colormap name (default: hot)
    --frame        Video frame index to visualize (per-frame / full-grid masks)
    --force        Visualize even if the directory mixes multiple runs
"""

import os
import re
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image


MASK_FILE_RE = re.compile(r"mask_step_(\d+)(?:_f_(\w+))?_t_(\d+)\.npy")


def _parse_mask_files(mask_dir: str) -> list[tuple[int, str | None, int, str]]:
    """Parse mask filenames into (step, frame_label, t, filename).

    frame_label is None for the averaged per-step masks, the frame number
    (e.g. "03") for per-frame masks, or "all" for full [f, h, w] grids.
    """
    out = []
    for f in sorted(os.listdir(mask_dir)):
        m = MASK_FILE_RE.match(f)
        if not m:
            continue
        out.append((int(m.group(1)), m.group(2), int(m.group(3)), f))
    return out


def load_masks(mask_dir: str) -> list[tuple[int, str | None, int, np.ndarray]]:
    """Load mask files, returning (step, frame_label, timestep, array)."""
    out = []
    for step, frame_label, t_val, f in _parse_mask_files(mask_dir):
        arr = np.load(os.path.join(mask_dir, f))
        out.append((step, frame_label, t_val, arr))
    return out


def check_masks_consistent(mask_dir: str) -> list[str]:
    """Detect a mask directory contaminated by more than one inference run.

    A single run writes exactly one mask per (step, frame) with timesteps
    strictly decreasing as the step grows. Duplicate (step, frame) files,
    mixed mask modes, or non-decreasing timesteps mean the directory mixes
    schedules and the grid would be misleading. Returns a list of
    human-readable problems; an empty list means the directory is consistent.
    """
    entries = _parse_mask_files(mask_dir)
    problems: list[str] = []
    if not entries:
        return problems

    def mode_of(label: str | None) -> str:
        if label is None:
            return "frame_avg"
        if label == "all":
            return "full_grid"
        return "per_frame"

    modes = {mode_of(label) for _, label, _, _ in entries}
    if len(modes) > 1:
        problems.append(
            f"mixed mask modes ({sorted(modes)}) — files from more than one run"
        )
        return problems
    mode = modes.pop()

    # Duplicate (step, frame) files ⇒ more than one run mixed together.
    seen: dict[tuple[int, str | None], int] = defaultdict(int)
    for step, frame_label, _, _ in entries:
        seen[(step, frame_label)] += 1
    for (step, frame_label), count in sorted(seen.items()):
        if count > 1:
            problems.append(
                f"step {step:02d} frame {frame_label} has {count} masks — "
                f"the directory mixes files from more than one run"
            )
    if problems:
        return problems

    # Per-frame runs must record the same frame set on every step.
    if mode == "per_frame":
        frames_by_step: dict[int, set[str]] = defaultdict(set)
        for step, frame_label, _, _ in entries:
            frames_by_step[step].add(frame_label)
        if len({frozenset(v) for v in frames_by_step.values()}) > 1:
            problems.append(
                "steps carry different frame sets — incomplete/mixed per-frame run"
            )

    # Timesteps must strictly decrease as the step grows (checked per frame label).
    t_by_frame: dict[str | None, dict[int, int]] = defaultdict(dict)
    for step, frame_label, t_val, _ in entries:
        t_by_frame[frame_label][step] = t_val
    for frame_label, steps_t in sorted(t_by_frame.items(), key=lambda kv: str(kv[0])):
        ts = [steps_t[s] for s in sorted(steps_t)]
        for a, b in zip(ts, ts[1:]):
            if b >= a:
                problems.append(
                    f"timestep not strictly decreasing (t={a} then t={b}) "
                    f"for frame {frame_label}"
                )
                break

    return problems


def select_display_masks(
    masks: list[tuple[int, str | None, int, np.ndarray]],
    frame: int | None = None,
) -> list[tuple[int, np.ndarray]]:
    """Reduce parsed masks to the (t, arr) list shown in the grid/GIF.

    - frame_avg : one [h, w] map per step, as saved.
    - full_grid : each step stores [f, h, w]; use --frame if given, else average.
    - per_frame : one [h, w] map per (step, frame); --frame picks which frame
      to show across all steps.
    """
    if not masks:
        return []

    def mode_of(label: str | None) -> str:
        if label is None:
            return "frame_avg"
        if label == "all":
            return "full_grid"
        return "per_frame"

    modes = {mode_of(label) for _, label, _, _ in masks}
    if len(modes) > 1:
        raise SystemExit("mask files mix modes; check ras_masks/")
    mode = modes.pop()

    if mode == "frame_avg":
        out = {step: (t_val, arr) for step, label, t_val, arr in masks if label is None}
        return [out[s] for s in sorted(out)]

    if mode == "full_grid":
        out = {}
        for step, label, t_val, arr in masks:
            grid = arr if frame is None else arr[frame]
            out[step] = (t_val, grid)
        return [out[s] for s in sorted(out)]

    # per_frame
    if frame is None:
        raise SystemExit(
            "per-frame masks need --frame N (0-based video frame index) to pick "
            "which frame to visualize across all steps"
        )
    key = f"{frame:02d}"
    out = {
        step: (t_val, arr)
        for step, label, t_val, arr in masks if label == key
    }
    if not out:
        labels = sorted({label for _, label, _, _ in masks if label is not None})
        raise SystemExit(
            f"no masks for frame {frame} (frame labels present: {labels})"
        )
    return [out[s] for s in sorted(out)]


def upsample_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Upsample a patch-space mask [h, w] to pixel-space [target_h, target_w]."""
    img = Image.fromarray((mask * 255).astype(np.uint8))
    img = img.resize((target_w, target_h), Image.NEAREST)
    return np.array(img).astype(np.float32) / 255.0


def apply_heatmap(mask: np.ndarray, alpha: float = 0.6, cmap_name: str = "hot") -> np.ndarray:
    """Convert a float mask [H, W] (0..1) to an RGBA heatmap using matplotlib colormap."""
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name)
    colored = cmap(mask)  # RGBA [H, W, 4]
    colored[:, :, 3] = mask * alpha  # alpha channel proportional to selection
    return (colored * 255).astype(np.uint8)


def create_grid_image(
    masks: list[tuple[int, np.ndarray]],
    height: int,
    width: int,
    cols: int = 6,
    cmap: str = "hot",
    max_steps: int = 60,
) -> Image.Image:
    """Arrange all step heatmaps in a grid.

    If there are more than max_steps masks, subsample evenly to max_steps
    to keep the grid legible.
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Subsample evenly if too many steps
    if len(masks) > max_steps:
        n_orig = len(masks)
        step = n_orig / max_steps
        indices = [int(i * step) for i in range(max_steps)]
        masks = [masks[i] for i in indices]
        print(f"  (subsampled from {n_orig} to {max_steps} steps for grid)")

    n = len(masks)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    axes = np.atleast_1d(axes).flatten()

    for i, (t_val, mask) in enumerate(masks):
        upsampled = upsample_mask(mask, height, width)
        axes[i].imshow(upsampled, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        frac = mask.mean()
        axes[i].set_title(f"t={t_val} ({frac:.1%})", fontsize=8)
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("RAS Region Selection Masks (brighter = selected more often)", fontsize=11)
    plt.tight_layout()

    # Render to PIL Image via BytesIO (compatible with matplotlib >= 3.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def create_gif(
    masks: list[tuple[int, np.ndarray]],
    height: int,
    width: int,
    output_path: str,
    fps: int = 5,
    cmap: str = "hot",
    max_frames: int = 60,
):
    """Create an animated GIF cycling through mask heatmaps."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Subsample evenly if too many steps
    if len(masks) > max_frames:
        step = len(masks) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        masks = [masks[i] for i in indices]

    frames = []
    for t_val, mask in masks:
        upsampled = upsample_mask(mask, height, width)
        frac = mask.mean()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(upsampled, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"Step t={t_val}  ({frac:.1%} tokens selected)", fontsize=10)
        ax.axis("off")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        frame = Image.open(buf)
        frames.append(frame)
        plt.close(fig)

    duration = int(1000 / fps)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )
    print(f"GIF saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize RAS selection masks")
    parser.add_argument("--mask_dir", default="ras_masks", help="Directory with mask_*.npy files")
    parser.add_argument("--output", default="ras_masks_grid.png", help="Output grid image")
    parser.add_argument("--gif_output", default="ras_masks.gif", help="Output animated GIF")
    parser.add_argument("--height", type=int, default=480, help="Video height in pixels")
    parser.add_argument("--width", type=int, default=832, help="Video width in pixels")
    parser.add_argument("--fps", type=int, default=5, help="GIF frames per second")
    parser.add_argument("--cmap", default="hot", help="Matplotlib colormap (hot, viridis, plasma, etc.)")
    parser.add_argument("--force", action="store_true",
                        help="Visualize even if the mask directory mixes files from multiple runs")
    parser.add_argument("--frame", type=int, default=None,
                        help="Video frame index to visualize (per-frame / full-grid masks); "
                             "default 0 for per-frame, average for full-grid")
    args = parser.parse_args()

    if not os.path.isdir(args.mask_dir):
        print(f"Error: mask directory '{args.mask_dir}' not found.")
        print("Run RAS-Wan2.1-T2V-1.3B.py with enable_viz=True first.")
        return

    # Refuse to visualize a directory contaminated by more than one run: the
    # grid would interleave unrelated schedules and look like a real zigzag.
    problems = check_masks_consistent(args.mask_dir)
    if problems and not args.force:
        print("Error: mask directory is not from a single run — refusing to visualize.")
        print("The grid would silently interleave masks from multiple schedules.")
        print()
        for p in problems:
            print(f"  - {p}")
        print()
        print("Fix: delete the stale mask_*.npy files (or the whole directory), re-run")
        print("inference once, then run this script again. Use --force to visualize")
        print("the mixed directory anyway.")
        return
    if problems:
        print("Warning: --force set — visualizing an inconsistent mask directory:")
        for p in problems:
            print(f"  ! {p}")

    masks = load_masks(args.mask_dir)
    if not masks:
        print(f"No mask_*.npy files found in '{args.mask_dir}'.")
        return

    try:
        display = select_display_masks(masks, frame=args.frame)
    except SystemExit as e:
        print(e)
        return

    print(f"Loaded {len(masks)} mask files from '{args.mask_dir}'; "
          f"displaying {len(display)} step(s)")
    for t_val, mask in display:
        print(f"  t={t_val:6.0f}  shape={mask.shape}  selected={mask.mean()*100:.1f}%")

    # Grid image
    print(f"\nCreating grid image ({len(display)} steps)...")
    grid = create_grid_image(display, args.height, args.width, cmap=args.cmap)
    grid.save(args.output)
    print(f"Grid saved to: {args.output}")

    # Animated GIF
    print(f"Creating animated GIF...")
    create_gif(display, args.height, args.width, args.gif_output, fps=args.fps, cmap=args.cmap)


if __name__ == "__main__":
    main()
