"""
VBench evaluation: Original Wan2.1-T2V-1.3B vs RAS-Wan — all dimensions.

End-to-end pipeline:
  1. Generate videos for both models over VBench's standard prompt suite
     using identical noise (same seed) per prompt so the comparison is fair.
     - original_wan  → full inference (all tokens, all steps)
     - ras_wan       → RAS sparse inference (KV cache + ratio)
  2. Visualize intermediate denoising frames side-by-side with the final
     clean frame ("with corresponding clean frame showing aside"), decoded
     from the latents captured at selected steps.
  3. Run VBench evaluation (`vbench evaluate --videos_path <dir>
     --dimension <dim>`) once per requested dimension, per model.

Dimensions and folders
----------------------
VBench's 16 dimensions share 11 prompt folders (identical prompt lists, per
VBench master `dimension_to_folder.json`), so each unique prompt is generated
exactly once and every dimension mapped to that folder is evaluated against
the same videos — exactly the official `evaluate.sh` protocol:

    subject_consistency ← subject_consistency, motion_smoothness, dynamic_degree (72)
    scene               ← scene, background_consistency                        (86)
    overall_consistency ← overall_consistency, aesthetic_quality,
                          imaging_quality                                      (93)
    object_class        ← object_class                                        (79)
    multiple_objects    ← multiple_objects                                    (82)
    color               ← color                                               (85)
    spatial_relationship← spatial_relationship                                (84)
    temporal_style      ← temporal_style                                     (100)
    human_action        ← human_action                                       (100)
    temporal_flickering ← temporal_flickering                                 (75)
    appearance_style    ← appearance_style                                    (90)

    946 unique prompts total.

Output layout (matches VBench `vbench_standard` mode — flat files named
`{prompt_en}-{sample_index}.mp4` directly inside the folder):

    sampled_videos/
    ├── original_wan/
    │   ├── human_action/A person is riding a bike-0.mp4
    │   ├── scene/...
    │   └── ...
    └── ras_wan/
        └── ...

    viz/
    ├── original_wan/human_action/prompt_000/   montage_step_XX.png,
    └── ras_wan/...                             clean_frame_40.png,
                                                progression_strip.png

Usage
-----
Full official benchmark — all 16 dimensions, all prompts (1 sample):
    python examples/wanvideo/model_inference/vbench_eval_RAS_vs_Wan.py --dimensions all

Official VBench protocol (5 samples per prompt):
    python examples/wanvideo/model_inference/vbench_eval_RAS_vs_Wan.py --dimensions all --num_samples 5

A single dimension (smoke test first, viz on):
    python examples/wanvideo/model_inference/vbench_eval_RAS_vs_Wan.py --dimensions human_action --max_prompts 2

A few dimensions:
    python examples/wanvideo/model_inference/vbench_eval_RAS_vs_Wan.py --dimensions human_action,scene,temporal_flickering

Shard a big run across machines:
    python examples/wanvideo/model_inference/vbench_eval_RAS_vs_Wan.py --dimensions all --prompt_start 0  --max_prompts 200 --no_evaluate
    python examples/wanvideo/model_inference/vbench_eval_RAS_vs_Wan.py --dimensions all --prompt_start 200 --max_prompts 200 --no_evaluate
    ...
    (then re-run one machine with --no_evaluate removed to evaluate everything)

Notes
-----
* The prompt suite is loaded from the installed vbench package
  (vbench/VBench_full_info.json) — the same file the evaluator reads — so the
  generated filenames are guaranteed to match. --prompt_file overrides it.
  For human_action only, the script falls back to an embedded copy of the
  official 100-prompt list.
* Existing output videos are skipped unless --force is given, so interrupted
  or sharded runs resume without regenerating.
* VBench's published numbers average 5 samples/prompt; scores from
  --num_samples 1 are noisier and not directly comparable.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from diffsynth.utils.data import save_video
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.models.wan_video_dit import set_to_torch_norm, selection_mask_to_grid


# ═══════════════════════════════════════════════════════════════
# VBench dimension → prompt-folder mapping (from vbench master
# dimension_to_folder.json). Prompts are stored ONCE per folder and shared by
# the dimensions that map to it; this mirrors the official evaluate.sh.
# ═══════════════════════════════════════════════════════════════

DIMENSION_TO_FOLDER = {
    "subject_consistency": "subject_consistency",
    "background_consistency": "scene",
    "aesthetic_quality": "overall_consistency",
    "imaging_quality": "overall_consistency",
    "object_class": "object_class",
    "multiple_objects": "multiple_objects",
    "color": "color",
    "spatial_relationship": "spatial_relationship",
    "scene": "scene",
    "temporal_style": "temporal_style",
    "overall_consistency": "overall_consistency",
    "human_action": "human_action",
    "temporal_flickering": "temporal_flickering",
    "motion_smoothness": "subject_consistency",
    "dynamic_degree": "subject_consistency",
    "appearance_style": "appearance_style",
}
ALL_DIMENSIONS = list(DIMENSION_TO_FOLDER.keys())

FOLDER_PROMPT_COUNTS = {  # unique prompts per folder, from VBench master full_info
    "appearance_style": 90, "color": 85, "human_action": 100, "multiple_objects": 82,
    "object_class": 79, "overall_consistency": 93, "scene": 86,
    "spatial_relationship": 84, "subject_consistency": 72,
    "temporal_flickering": 75, "temporal_style": 100,
}


# ═══════════════════════════════════════════════════════════════
# Official VBench human_action prompts (from vbench/VBench_full_info.json,
# master branch). Used only as a fallback for the human_action folder when the
# installed vbench package cannot be located. Filenames must match these exactly.
# ═══════════════════════════════════════════════════════════════

VBENCH_HUMAN_ACTION_PROMPTS = [
    "A person is riding a bike",
    "A person is marching",
    "A person is roller skating",
    "A person is tasting beer",
    "A person is clapping",
    "A person is drawing",
    "A person is petting animal (not cat)",
    "A person is eating watermelon",
    "A person is playing harp",
    "A person is wrestling",
    "A person is riding scooter",
    "A person is sweeping floor",
    "A person is skateboarding",
    "A person is dunking basketball",
    "A person is playing flute",
    "A person is stretching leg",
    "A person is tying tie",
    "A person is skydiving",
    "A person is shooting goal (soccer)",
    "A person is playing piano",
    "A person is finger snapping",
    "A person is canoeing or kayaking",
    "A person is laughing",
    "A person is digging",
    "A person is clay pottery making",
    "A person is shooting basketball",
    "A person is bending back",
    "A person is shaking hands",
    "A person is bandaging",
    "A person is push up",
    "A person is catching or throwing frisbee",
    "A person is playing trumpet",
    "A person is flying kite",
    "A person is filling eyebrows",
    "A person is shuffling cards",
    "A person is folding clothes",
    "A person is smoking",
    "A person is tai chi",
    "A person is squat",
    "A person is playing controller",
    "A person is throwing axe",
    "A person is giving or receiving award",
    "A person is air drumming",
    "A person is taking a shower",
    "A person is planting trees",
    "A person is sharpening knives",
    "A person is robot dancing",
    "A person is rock climbing",
    "A person is hula hooping",
    "A person is writing",
    "A person is bungee jumping",
    "A person is pushing cart",
    "A person is cleaning windows",
    "A person is cutting watermelon",
    "A person is cheerleading",
    "A person is washing hands",
    "A person is ironing",
    "A person is cutting nails",
    "A person is hugging",
    "A person is trimming or shaving beard",
    "A person is jogging",
    "A person is making bed",
    "A person is washing dishes",
    "A person is grooming dog",
    "A person is doing laundry",
    "A person is knitting",
    "A person is reading book",
    "A person is baby waking up",
    "A person is massaging legs",
    "A person is brushing teeth",
    "A person is crawling baby",
    "A person is motorcycling",
    "A person is driving car",
    "A person is sticking tongue out",
    "A person is shaking head",
    "A person is sword fighting",
    "A person is doing aerobics",
    "A person is strumming guitar",
    "A person is riding or walking with horse",
    "A person is archery",
    "A person is catching or throwing baseball",
    "A person is playing chess",
    "A person is rock scissors paper",
    "A person is using computer",
    "A person is arranging flowers",
    "A person is bending metal",
    "A person is ice skating",
    "A person is climbing a rope",
    "A person is crying",
    "A person is dancing ballet",
    "A person is getting a haircut",
    "A person is running on treadmill",
    "A person is kissing",
    "A person is counting money",
    "A person is barbequing",
    "A person is peeling apples",
    "A person is milking cow",
    "A person is shining shoes",
    "A person is making snowman",
    "A person is sailing",
]


NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# ═══════════════════════════════════════════════════════════════
# Prompt plan resolution
# ═══════════════════════════════════════════════════════════════

def _entry_dims(entry):
    if isinstance(entry, dict):
        dims = entry.get("dimension", [])
        if isinstance(dims, str):
            dims = [dims]
        return dims
    return []


def _entry_prompt(entry):
    if isinstance(entry, dict):
        return entry.get("prompt_en")
    return entry if isinstance(entry, str) else None


def load_full_info(args):
    """Load the full VBench prompt info.

    Returns (entries, full_info_path_or_None). `entries` is a list whose
    elements are either dicts {dimension, prompt_en} (full_info style) or plain
    prompt strings (a single-dimension .txt list). Priority: --prompt_file,
    then the installed vbench package's VBench_full_info.json.
    """
    if args.prompt_file:
        path = os.path.abspath(args.prompt_file)
        if path.lower().endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = list(data.values())
            print(f"Loaded prompt info from {path}")
            return data, path
        lines = [ln.strip() for ln in open(path) if ln.strip()]
        print(f"Loaded single-dimension prompt list from {path}")
        return lines, None

    try:
        import vbench
        pkg_dir = os.path.dirname(os.path.abspath(vbench.__file__))
    except ImportError:
        pkg_dir = None

    if pkg_dir:
        cand = os.path.join(pkg_dir, "VBench_full_info.json")
        if os.path.isfile(cand):
            with open(cand) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = list(data.values())
            print(f"Loaded prompt info from VBench package:\n  {cand}")
            return data, cand
        print(f"  (no VBench_full_info.json under {pkg_dir})")
    return None, None


def parse_dimensions(arg: str) -> list[str]:
    if arg is None or arg.lower() == "all":
        return list(ALL_DIMENSIONS)
    dims = [d.strip() for d in str(arg).split(",") if d.strip()]
    unknown = [d for d in dims if d not in DIMENSION_TO_FOLDER]
    if unknown:
        raise SystemExit(f"Unknown dimension(s): {unknown}. Valid dimensions:\n"
                         f"  {', '.join(ALL_DIMENSIONS)}")
    return dims


def build_plan(dims, entries, args):
    """Build a generation plan.

    Returns {folder: {"members": [dims...], "prompts": [(folder_idx, global_idx, prompt)...]}}.
    Only the folders of the requested dimensions are included; prompts within a
    shared folder are generated once and serve every dimension mapped to it.
    folder_idx is the prompt's position within its folder's FULL list (0..n-1) and
    is what seeds depend on, so sharded runs (--prompt_start/--max_prompts) use
    identical noise to a full run. global_idx is its position in the prompt source
    (for stable per-prompt directory names).
    """
    # Plain prompt list (e.g. a .txt) → single-dimension mode.
    if entries is not None and all(not _entry_dims(e) for e in entries):
        if len(dims) != 1:
            raise SystemExit("--prompt_file with a plain prompt list requires exactly "
                             "one --dimensions value (the list has no dimension info).")
        folder = DIMENSION_TO_FOLDER[dims[0]]
        return {folder: {"members": list(dims),
                         "prompts": [(i, i, str(e)) for i, e in enumerate(entries)]}}

    plan = {}
    folders = sorted({DIMENSION_TO_FOLDER[d] for d in dims})
    for folder in folders:
        members = [d for d in dims if DIMENSION_TO_FOLDER[d] == folder]
        prompts = []
        if entries is not None:
            for g, e in enumerate(entries):
                if _entry_prompt(e) and any(m in _entry_dims(e) for m in members):
                    prompts.append((len(prompts), g, _entry_prompt(e)))
        if not prompts:
            if folder == "human_action":
                print(f"  (folder '{folder}': no source prompts found — using the embedded "
                      f"official human_action list)")
                prompts = [(i, i, p) for i, p in enumerate(VBENCH_HUMAN_ACTION_PROMPTS)]
            else:
                raise SystemExit(
                    f"No prompt source for folder '{folder}'. Ensure vbench is installed "
                    f"(vbench/VBench_full_info.json) or pass --prompt_file.")
        plan[folder] = {"members": members, "prompts": prompts}
    return plan


# ═══════════════════════════════════════════════════════════════
# Model / prompt helpers
# ═══════════════════════════════════════════════════════════════

def encode_prompt(pipe, prompt: str) -> torch.Tensor:
    """Encode a text prompt with the pipeline's T5 tokenizer + text encoder."""
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


def geometry(pipe, num_frames: int, height: int, width: int) -> dict:
    """Latent + patch geometry shared by both models."""
    vae = pipe.vae
    dit = pipe.dit
    latent_frames = (num_frames - 1) // 4 + 1          # temporal compression
    latent_h = height // vae.upsampling_factor          # spatial compression
    latent_w = width // vae.upsampling_factor
    # Patchify reduces spatial dims by patch_size[1:]. S is tokens AFTER patchify.
    S = latent_frames * (latent_h // dit.patch_size[1]) * (latent_w // dit.patch_size[2])
    return {
        "shape": (1, vae.model.z_dim, latent_frames, latent_h, latent_w),
        "S": S,
    }


# ═══════════════════════════════════════════════════════════════
# Denoising
# ═══════════════════════════════════════════════════════════════

def run_denoise(dit, scheduler, latents, ctx_posi, ctx_nega, *,
                cfg_scale, mode, ratio, num_dense_steps, extra_dense_steps,
                dumb_update, S, dtype, device, capture_steps, enable_masks):
    """Run the full denoising loop.

    mode='full' → all tokens every step (kv_cache=None).
    mode='ras'  → RAS sparse updates with per-layer KV caches.

    Returns (final_latents, captured, elapsed_s), where
    captured = {step_index: (t_val, latents_cpu)} for step_index in capture_steps.
    """
    B = latents.shape[0]
    num_layers = len(dit.blocks)
    all_patches = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)

    if mode == "ras":
        kv_posi = [{} for _ in range(num_layers)]
        ctx_posi_cache = [{} for _ in range(num_layers)]
        kv_nega = [{} for _ in range(num_layers)]
        ctx_nega_cache = [{} for _ in range(num_layers)]
        skip_list = torch.zeros(B, S, device=device)
        skip_k = torch.zeros(B, S, device=device)
    else:
        kv_posi = kv_nega = None
        ctx_posi_cache = ctx_nega_cache = None
        skip_list = skip_k = None

    dit._prev_noise_tokens = None
    if enable_masks:
        dit.clear_selection_masks()

    captured = {}
    timesteps = scheduler.timesteps
    t0 = time.perf_counter()

    # torch.inference_mode() avoids the autograd overhead that caused OOM on
    # multi-GB intermediate activations during long runs.
    with torch.inference_mode():
        for i, ts in enumerate(timesteps):
            t = ts.unsqueeze(0).to(dtype=dtype, device=device)

            ras_active = mode == "ras"
            is_dense = i < num_dense_steps or i in extra_dense_steps
            selected = all_patches if (ras_active and is_dense) else None

            # --- Positive (conditional) forward ---
            noise_posi = dit.forward(
                x=latents,
                timestep=t,
                context=ctx_posi,
                kv_cache=kv_posi,
                ctx_kv_cache=ctx_posi_cache,
                skip_list=skip_list,
                skip_k=skip_k,
                selected_patches=selected,
                ratio=ratio if ras_active else 1.0,
                dumb_update=dumb_update,
                enable_debug_masks=(ras_active and enable_masks),
            )
            torch.cuda.empty_cache()

            # --- Classifier-free guidance ---
            if cfg_scale != 1.0:
                # Negative branch must process the SAME tokens as the positive
                # branch so the skip record updates once per step. On sparse
                # steps pass the positive branch's selection explicitly.
                nega_selected = (
                    dit.get_last_selected_patches()
                    if (ras_active and selected is None) else selected
                )
                noise_nega = dit.forward(
                    x=latents,
                    timestep=t,
                    context=ctx_nega,
                    kv_cache=kv_nega,
                    ctx_kv_cache=ctx_nega_cache,
                    skip_list=skip_list,
                    skip_k=skip_k,
                    selected_patches=nega_selected,
                    ratio=ratio if ras_active else 1.0,
                    dumb_update=dumb_update,
                    enable_debug_masks=False,
                )
                noise_pred = noise_nega + cfg_scale * (noise_posi - noise_nega)
            else:
                noise_pred = noise_posi

            if i in capture_steps:
                captured[i] = (float(ts.item()), latents.detach().cpu().clone())

            latents = scheduler.step(noise_pred, timesteps[i], latents)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return latents, captured, elapsed


def decode_video(pipe, vae, latents, device) -> list:
    """Decode latents to a list of PIL frames (T)."""
    video = vae.decode(
        latents, device=device,
        tiled=True, tile_size=(30, 52), tile_stride=(15, 26),
    )
    return pipe.vae_output_to_video(video)


# ═══════════════════════════════════════════════════════════════
# Intermediate-frame visualization
# ═══════════════════════════════════════════════════════════════

def find_font(size: int = 26):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ):
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def build_montage(left, right, left_label, right_label, out_path):
    """Side-by-side [intermediate | clean] image with labels above each side."""
    font = find_font()
    w, h = left.size
    label_h = max(font.size + 14, 30)
    gap = 12
    canvas = Image.new("RGB", (w * 2 + gap * 3, h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 8), left_label, fill="black", font=font)
    draw.text((w + gap * 2, 8), right_label, fill="black", font=font)
    canvas.paste(left, (gap, label_h))
    canvas.paste(right, (w + gap * 2, label_h))
    canvas.save(out_path)


def build_progression_strip(entries, clean_frame, out_path, thumb_h=256):
    """Horizontal strip of all intermediate frames followed by the clean frame."""
    font = find_font()
    label_h = font.size + 14

    def thumb(img):
        scale = thumb_h / img.size[1]
        return img.resize((int(img.size[0] * scale), thumb_h))

    thumbs = [(f"step {s} t={int(t)}", thumb(img)) for s, t, img in entries]
    thumbs.append(("clean (final)", thumb(clean_frame)))

    tw = sum(t.size[0] for _, t in thumbs) + 16 * (len(thumbs) + 1)
    canvas = Image.new("RGB", (tw, thumb_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    x = 8
    for label, t in thumbs:
        draw.text((x, 6), label, fill="black", font=font)
        canvas.paste(t, (x, label_h))
        x += t.size[0] + 16
    canvas.save(out_path)


def save_intermediate_viz(out_dir, captured, clean_frame, viz_frame,
                          dtype, device, pipe, vae):
    """Decode captured intermediate latents and save per-step montages + strip."""
    os.makedirs(out_dir, exist_ok=True)
    clean_path = os.path.join(out_dir, f"clean_frame_{viz_frame:02d}.png")
    clean_frame.save(clean_path)

    entries = []
    for step in sorted(captured):
        t_val, lat = captured[step]
        lat = lat.to(dtype=dtype, device=device)
        frames = decode_video(pipe, vae, lat, device)
        inter = frames[viz_frame]

        frame_path = os.path.join(
            out_dir, f"inter_step_{step:02d}_t_{t_val:04.0f}_frame{viz_frame:02d}.png")
        inter.save(frame_path)

        montage_path = os.path.join(out_dir, f"montage_step_{step:02d}.png")
        build_montage(
            inter, clean_frame,
            f"intermediate @ step {step} (t={t_val:.0f})", "clean (final)",
            montage_path,
        )
        entries.append((step, t_val, inter))

    build_progression_strip(
        entries, clean_frame, os.path.join(out_dir, "progression_strip.png"))
    return out_dir


def save_ras_masks(dit, out_dir):
    """Save per-step selection masks (frame-average) for visualize_ras_masks.py."""
    masks = dit.get_selection_masks()
    os.makedirs(out_dir, exist_ok=True)
    for idx, (t_val, mask, (f, h, w)) in enumerate(masks):
        grid = selection_mask_to_grid(mask, (f, h, w))  # [1, 1, f, h, w]
        frame_avg = grid[0, 0].mean(dim=0).cpu().numpy()  # [h, w]
        np.save(os.path.join(out_dir, f"mask_step_{idx:02d}_t_{int(t_val):d}.npy"), frame_avg)
    print(f"  RAS masks saved ({len(masks)} steps) -> {out_dir}")


# ═══════════════════════════════════════════════════════════════
# VBench evaluation
# ═══════════════════════════════════════════════════════════════

def run_vbench_evaluate(model, dimension, videos_dir, args, full_info_path):
    if shutil.which("vbench") is None:
        print("  ERROR: 'vbench' command not found on PATH. Install VBench "
              "(pip install vbench) or pass --no_evaluate.")
        return -1
    cmd = [
        "vbench", "evaluate",
        "--videos_path", videos_dir,
        "--dimension", dimension,
        "--output_path", os.path.join(args.eval_output_root, model),
        "--ngpus", str(args.ngpus),
    ]
    if full_info_path:
        cmd += ["--full_json_dir", full_info_path]
    print(f"\n  Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"  [vbench evaluate] FAILED (exit {proc.returncode}) after {elapsed:.0f}s")
    else:
        print(f"  [vbench evaluate] done in {elapsed:.0f}s")
    return proc.returncode


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="VBench evaluation for original Wan vs RAS-Wan (all dimensions).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # smoke test (1 dimension, 2 prompts, viz on)\n"
            "  python examples/wanvideo/model_inference/vbench_eval_RAS_vs_Wan.py --dimensions human_action --max_prompts 2\n"
            "  # full official benchmark (all 16 dimensions, all 946 prompts)\n"
            "  python examples/wanvideo/model_inference/vbench_eval_RAS_vs_Wan.py --dimensions all\n"
            "  # official VBench protocol (5 samples per prompt)\n"
            "  python examples/wanvideo/model_inference/vbench_eval_RAS_vs_Wan.py --dimensions all --num_samples 5\n"
        ),
    )

    # What / where
    parser.add_argument("--model_id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--dimensions", default="all",
                        help="Comma-separated VBench dimensions to evaluate, or 'all' "
                             f"(all 16: {', '.join(ALL_DIMENSIONS)}).")
    parser.add_argument("--dimension", dest="dimension_alias", default=None,
                        help=argparse.SUPPRESS)  # singular alias, as in the vbench CLI
    parser.add_argument("--prompt_file", default=None,
                        help="Override prompt source: a VBench_full_info.json, a "
                             "prompts_per_dimension .txt (single dimension), or a "
                             "JSON {key: prompt} dict.")
    parser.add_argument("--prompt_start", type=int, default=0,
                        help="First prompt index to generate per folder (for sharding).")
    parser.add_argument("--max_prompts", type=int, default=0,
                        help="Number of prompts to generate per folder (0 = all).")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Per-prompt sample count. VBench's official protocol uses 5.")
    parser.add_argument("--videos_root", default="sampled_videos",
                        help="Root that receives {videos_root}/{model}/{folder}/.")
    parser.add_argument("--model_names", default="original_wan,ras_wan",
                        help="Comma-separated model folder names. Names containing 'ras' "
                             "use the RAS sparse path; others use full inference.")
    parser.add_argument("--eval_output_root", default="evaluation_results")
    parser.add_argument("--ngpus", type=int, default=1, help="GPUs for vbench evaluate.")
    parser.add_argument("--no_evaluate", action="store_true", help="Skip the vbench step.")

    # Generation
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; per-sample seed = seed + prompt_idx*num_samples + sample "
                             "(prompt_idx is the prompt's index in the full prompt suite, so seeds "
                             "are stable across sharded runs).")
    parser.add_argument("--negative_prompt", default=NEGATIVE_PROMPT)

    # RAS
    parser.add_argument("--ratio", type=float, default=0.25,
                        help="Fraction of tokens processed per sparse step.")
    parser.add_argument("--num_dense_steps", type=int, default=20,
                        help="Initial full-update steps that warm the KV caches.")
    parser.add_argument("--extra_dense_steps", default="30,40",
                        help="Comma-separated step indices also forced dense (matches the "
                             "current RAS-Wan2.1-T2V-1.3B.py configuration).")
    parser.add_argument("--dumb_update", choices=["Previous", "Zero"], default="Previous")

    # Visualization
    parser.add_argument("--no_viz", action="store_true",
                        help="Disable intermediate-frame visualization.")
    parser.add_argument("--viz_prompts", type=int, default=1,
                        help="How many prompts per folder get the full visualization treatment.")
    parser.add_argument("--viz_steps", default="5,10,20,30,40",
                        help="Comma-separated denoising step indices at which to capture "
                             "intermediate latents.")
    parser.add_argument("--viz_frame", type=int, default=None,
                        help="Frame index shown in the montages (default: middle frame).")
    parser.add_argument("--viz_dir", default="viz")
    parser.add_argument("--save_masks", action="store_true",
                        help="Also save RAS selection masks for visualize_ras_masks.py.")

    # Misc
    parser.add_argument("--keep_vae_on_gpu", action="store_true",
                        help="Leave the VAE on GPU between decodes (faster, more VRAM). "
                             "Default offloads the VAE to CPU between prompts, matching "
                             "RAS-Wan2.1-T2V-1.3B.py.")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate videos even if the output file already exists.")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Prompt plan ────────────────────────────────────────────────
    dims = parse_dimensions(args.dimension_alias or args.dimensions)
    entries, full_info_path = load_full_info(args)
    plan = build_plan(dims, entries, args)

    # Slice per folder for sharding.
    start = args.prompt_start
    selected_plan = {}
    for folder, info in plan.items():
        ps = info["prompts"]
        sel = ps[start:] if args.max_prompts <= 0 else ps[start:start + args.max_prompts]
        selected_plan[folder] = {"members": info["members"], "prompts": sel}

    models = [m.strip() for m in args.model_names.split(",") if m.strip()]
    if not models:
        raise SystemExit("--model_names is empty.")

    n_prompts = sum(len(i["prompts"]) for i in selected_plan.values())
    num_videos = n_prompts * args.num_samples * len(models)

    print("=" * 72)
    print("VBench evaluation: original Wan vs RAS-Wan")
    print("=" * 72)
    print(f"  dimensions         : {len(dims)}  ({', '.join(dims)})")
    print(f"  folders            : {len(selected_plan)}  (shared prompt sets)")
    for folder, info in selected_plan.items():
        full = FOLDER_PROMPT_COUNTS.get(folder, "?")
        print(f"      {folder:<22} {len(info['prompts']):>3}/{full} prompts "
              f"(dims: {', '.join(info['members'])})")
    print(f"  samples / prompt   : {args.num_samples}  (VBench official = 5)")
    print(f"  models             : {', '.join(models)}")
    print(f"  videos to generate : {num_videos}")
    if args.num_samples != 5:
        print("  NOTE: --num_samples != 5 → scores are not directly comparable to")
        print("        published VBench numbers (which average 5 samples/prompt).")
    if args.max_prompts != 0 or args.prompt_start != 0:
        print("  NOTE: sharded/partial run — reported scores are partial approximations")
        print("        over the prompts generated so far.")
    if args.prompt_start == 0 and args.max_prompts == 0 and num_videos > 1000:
        print("  NOTE: this is the full 946-prompt benchmark (~2k videos for 2 models).")
        print("        Prefer sharding with --prompt_start/--max_prompts --no_evaluate,")
        print("        or a smoke test: --dimensions human_action --max_prompts 2.")
    print()

    # ── Model ────────────────────────────────────────────────────────
    print("Loading model...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="google/umt5-xxl/"),
    )
    dit = pipe.dit
    vae = pipe.vae
    scheduler = pipe.scheduler
    device = pipe.device
    dtype = pipe.torch_dtype

    print(f"  DiT blocks: {len(dit.blocks)}, dim: {dit.dim}, patch_size: {dit.patch_size}")
    set_to_torch_norm([dit])
    print("  RMSNorm: torch_norm enabled")

    scheduler.set_timesteps(args.num_inference_steps, denoising_strength=1.0, shift=5.0)
    geo = geometry(pipe, args.num_frames, args.height, args.width)
    print(f"  Latent shape: {geo['shape']}, tokens S={geo['S']}")

    # ── Visualization setup ──────────────────────────────────────────
    viz_frame = args.viz_frame if args.viz_frame is not None else args.num_frames // 2
    if not args.no_viz:
        viz_steps = {int(x) for x in args.viz_steps.split(",") if x.strip()}
        capture_steps = {s for s in viz_steps if 0 <= s < args.num_inference_steps}
        if viz_steps - capture_steps:
            print(f"  (clamped capture steps to [0, {args.num_inference_steps}))")
    else:
        capture_steps = set()

    # ── Generate ─────────────────────────────────────────────────────
    total_time = {m: 0.0 for m in models}
    n_done = {m: 0 for m in models}

    for folder, info in selected_plan.items():
        print(f"\n===== Folder: {folder}  (dimensions: {', '.join(info['members'])}) =====")
        print(f"Encoding {len(info['prompts'])} prompts for this folder...")

        # Encode this folder's prompts up front, then drop the (large) T5 encoder.
        ctx_posi_all = [encode_prompt(pipe, p) for _, _, p in info["prompts"]]
        ctx_nega = encode_prompt(pipe, args.negative_prompt)
        pipe.text_encoder.to("cpu")
        if not args.keep_vae_on_gpu:
            vae.to("cpu")
        torch.cuda.empty_cache()

        for j, (folder_idx, gidx, prompt) in enumerate(tqdm(info["prompts"], desc=f"folder {folder}")):
            ctx_posi = ctx_posi_all[j]
            print(f"\nPrompt {folder_idx} (source #{gidx}): {prompt}")

            for model in models:
                mode = "ras" if "ras" in model else "full"
                out_dir = os.path.join(args.videos_root, model, folder)
                os.makedirs(out_dir, exist_ok=True)
                do_viz = bool(capture_steps) and j < args.viz_prompts

                for s in range(args.num_samples):
                    seed = args.seed + folder_idx * args.num_samples + s
                    vid_path = os.path.join(out_dir, f"{prompt}-{s}.mp4")

                    if os.path.exists(vid_path) and not args.force:
                        print(f"  [{model}/{s}] skip (exists): {vid_path}")
                        continue

                    latents = pipe.generate_noise(geo["shape"], seed=seed, rand_device="cpu")
                    latents = latents.to(dtype=dtype, device=device)

                    final, captured, elapsed = run_denoise(
                        dit, scheduler, latents, ctx_posi, ctx_nega,
                        cfg_scale=args.cfg_scale,
                        mode=mode,
                        ratio=args.ratio,
                        num_dense_steps=args.num_dense_steps,
                        extra_dense_steps={int(x) for x in args.extra_dense_steps.split(",") if x.strip()},
                        dumb_update=args.dumb_update,
                        S=geo["S"],
                        dtype=dtype,
                        device=device,
                        capture_steps=capture_steps if do_viz else set(),
                        enable_masks=bool(args.save_masks and mode == "ras"),
                    )

                    # Decode final video (VAE on GPU)
                    vae.to(device)
                    frames = decode_video(pipe, vae, final, device)
                    save_video(frames, vid_path, fps=args.fps, quality=5)

                    # Intermediate-frame visualization (VAE still on GPU)
                    if do_viz and captured:
                        clean = frames[viz_frame]
                        viz_out = os.path.join(args.viz_dir, model, folder, f"prompt_{gidx:03d}")
                        save_intermediate_viz(viz_out, captured, clean, viz_frame,
                                              dtype, device, pipe, vae)
                        print(f"  intermediate-frame montages -> {viz_out}/")

                    if args.save_masks and mode == "ras":
                        masks_out = os.path.join(args.viz_dir, model, folder, f"prompt_{gidx:03d}", "masks")
                        save_ras_masks(dit, masks_out)

                    if not args.keep_vae_on_gpu:
                        vae.to("cpu")
                    torch.cuda.empty_cache()

                    total_time[model] += elapsed
                    n_done[model] += 1
                    print(f"  [{model}/{s}] {elapsed:6.1f}s  {elapsed / args.num_inference_steps:5.1f} ms/step  -> {vid_path}")

    # ── Report ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("GENERATION SUMMARY")
    print("=" * 72)
    for model in models:
        if n_done[model]:
            avg = total_time[model] / n_done[model]
            print(f"  {model:<14s} {n_done[model]:>3d} videos  total {total_time[model]:7.1f}s  "
                  f"avg {avg:6.1f}s/video")
        else:
            print(f"  {model:<14s} 0 videos (all skipped)")
    print(f"\n  Videos:  {os.path.abspath(args.videos_root)}/<model>/<folder>/")
    print(f"  Montages:{os.path.abspath(args.viz_dir)}/")

    # ── VBench evaluation ────────────────────────────────────────────
    if args.no_evaluate:
        print("\nSkipping VBench evaluation (--no_evaluate). Run it manually, e.g.:")
        for model in models:
            for dim in dims:
                folder = DIMENSION_TO_FOLDER[dim]
                print(f"    vbench evaluate --videos_path \"{args.videos_root}/{model}/{folder}\" "
                      f"--dimension {dim}")
        return

    print("\n" + "=" * 72)
    print("VBench EVALUATION")
    print("=" * 72)
    failures = 0
    for model in models:
        for dim in dims:
            folder = DIMENSION_TO_FOLDER[dim]
            videos_dir = os.path.abspath(os.path.join(args.videos_root, model, folder))
            rc = run_vbench_evaluate(model, dim, videos_dir, args, full_info_path)
            failures += 1 if rc != 0 else 0
            print(f"  Results: {os.path.abspath(args.eval_output_root)}/{model}/")

    if failures:
        sys.exit(f"{failures} VBench evaluation(s) failed — see output above.")


if __name__ == "__main__":
    main()
