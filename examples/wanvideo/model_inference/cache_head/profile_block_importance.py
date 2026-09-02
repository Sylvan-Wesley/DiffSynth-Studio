"""Rank the teacher's DiT blocks by how much they actually change the residual stream.

Once self-attention is sparse the FFN dominates each block's cost, so depth is
the main remaining lever on a head step.  Dropping blocks by uniform stride is a
guess; this measures instead.

Each ``DiTBlock`` is a residual update, so a block's influence is how far it
moves its own input:

    relative_delta = ||block(x) - x|| / ||x||
    cosine         = cos(block(x), x)

A block with a small relative delta and a cosine near 1 is nearly the identity
and is the cheapest thing to remove.  This is the ShortGPT / block-influence
criterion, applied to the actual prompts and timesteps the student will see.

Usage::

    python profile_block_importance.py --captions mixkit_captions.jsonl \\
        --num-prompts 8 --keep 15 --out-dir block_importance

The suggested indices are printed in the exact form
``--student-layer-indices`` expects.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


# ═══════════════════════════════════════════════════════════════
# Selection (pure, unit-testable)
# ═══════════════════════════════════════════════════════════════

def select_important_layers(scores: list[float], num_layers: int) -> tuple[int, ...]:
    """Keep the ``num_layers`` most influential blocks, always retaining the
    endpoints.

    The first and last blocks are pinned regardless of score: they read the
    patch embedding and feed the output head, and dropping either changes what
    the inherited weights on both sides are looking at.
    """
    total = len(scores)
    if total == 0:
        raise ValueError("scores must be non-empty")
    if not 1 <= num_layers <= total:
        raise ValueError(f"num_layers must be in [1, {total}], got {num_layers}")
    if num_layers == total:
        return tuple(range(total))
    if num_layers == 1:
        return (0,)

    pinned = {0, total - 1}
    remaining = num_layers - len(pinned)
    candidates = sorted(
        (i for i in range(total) if i not in pinned),
        key=lambda i: scores[i],
        reverse=True,
    )
    return tuple(sorted(pinned | set(candidates[:remaining])))


def summarize(records: list[dict], num_layers: int) -> list[dict]:
    """Average the per-(prompt, step) measurements down to one row per block."""
    totals: dict[int, dict[str, float]] = {
        layer: {"relative_delta": 0.0, "cosine": 0.0, "count": 0.0} for layer in range(num_layers)
    }
    for record in records:
        bucket = totals[record["layer"]]
        bucket["relative_delta"] += record["relative_delta"]
        bucket["cosine"] += record["cosine"]
        bucket["count"] += 1.0
    rows = []
    for layer in range(num_layers):
        bucket = totals[layer]
        count = max(bucket["count"], 1.0)
        rows.append({
            "layer": layer,
            "relative_delta": bucket["relative_delta"] / count,
            "cosine": bucket["cosine"] / count,
            "samples": int(bucket["count"]),
        })
    return rows


# ═══════════════════════════════════════════════════════════════
# Probe
# ═══════════════════════════════════════════════════════════════

class BlockContributionProbe:
    """Wrap every ``DiTBlock.forward`` to record how far it moves its input.

    Uses the same monkeypatch-a-submodule technique as
    ``visualize_wan_teacher_attention.py``: the wrapper is a pure side channel
    and returns the original output untouched, so the trajectory is unaffected.
    """

    def __init__(self, dit):
        self.dit = dit
        self.records: list[dict] = []
        self.step = 0
        self.enabled = False
        self._originals: list = []

    def install(self) -> None:
        for layer_index, block in enumerate(self.dit.blocks):
            original = block.forward

            def wrapped(x, *args, _original=original, _layer=layer_index, **kwargs):
                out = _original(x, *args, **kwargs)
                if self.enabled:
                    self._record(_layer, x, out)
                return out

            self._originals.append((block, original))
            block.forward = wrapped

    def restore(self) -> None:
        for block, original in self._originals:
            block.forward = original
        self._originals = []

    @torch.no_grad()
    def _record(self, layer: int, x: torch.Tensor, out: torch.Tensor) -> None:
        a = x.detach().float().flatten(1)
        b = out.detach().float().flatten(1)
        delta = (b - a).norm(dim=-1)
        base = a.norm(dim=-1).clamp_min(1e-12)
        cosine = torch.nn.functional.cosine_similarity(a, b, dim=-1)
        self.records.append({
            "layer": layer,
            "step": self.step,
            "relative_delta": float((delta / base).mean()),
            "cosine": float(cosine.mean()),
        })


# ═══════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════

def run_profile(args: argparse.Namespace) -> None:
    from diffsynth import ModelConfig, WanVideoPipeline
    from diffsynth.models.wan_video_dit import set_to_torch_norm

    from cache_head_model import CacheHeadSchedule, parse_full_step_indices  # noqa: F401
    from cache_head_model_inference import full_step
    from cache_head_model_training import PromptDataset

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
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

    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.num_steps, denoising_strength=1.0, shift=args.sigma_shift)

    def encode(caption: str) -> torch.Tensor:
        ids, mask = pipe.tokenizer(
            [caption], return_tensors="pt", padding="max_length", truncation=True, max_length=512
        ).values()
        ids, mask = ids.to(device), mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            emb[i, v:] = 0
        return emb.detach()

    neg_ctx = encode(args.negative_prompt)

    dataset = PromptDataset(args.captions, split=args.split, subset=args.num_prompts)
    captions = [caption for _, caption in list(dataset)[: args.num_prompts]]
    if not captions:
        raise SystemExit(f"no captions in split {args.split!r} of {args.captions}")

    z_dim = pipe.vae.model.z_dim
    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_h = args.height // pipe.vae.upsampling_factor
    latent_w = args.width // pipe.vae.upsampling_factor
    latent_shape = (1, z_dim, latent_frames, latent_h, latent_w)

    probe = BlockContributionProbe(dit)
    probe.install()
    try:
        for prompt_index, caption in enumerate(captions):
            ctx = encode(caption)
            generator = torch.Generator(device="cpu").manual_seed(args.seed + prompt_index)
            latents = torch.randn(latent_shape, generator=generator).to(device=device, dtype=dtype)
            print(f"[{prompt_index + 1}/{len(captions)}] {caption[:70]}")
            for step in range(args.num_steps):
                timestep = scheduler.timesteps[step].reshape(1).to(device=device, dtype=dtype)
                probe.step = step
                probe.enabled = True
                noise_pred, _ = full_step(dit, latents, timestep, ctx, neg_ctx, args.cfg_scale)
                probe.enabled = False
                latents = scheduler.step(noise_pred, timestep, latents)
    finally:
        probe.restore()

    num_layers = len(dit.blocks)
    rows = summarize(probe.records, num_layers)
    scores = [row["relative_delta"] for row in rows]
    keep = select_important_layers(scores, args.keep) if args.keep else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": args.model_id,
        "num_layers": num_layers,
        "num_prompts": len(captions),
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "per_layer": rows,
        "suggested_layer_indices": list(keep) if keep else None,
    }
    (out_dir / "block_importance.json").write_text(json.dumps(payload, indent=2))

    print(f"\n{'layer':>5}  {'rel delta':>10}  {'cosine':>8}   rank")
    order = sorted(range(num_layers), key=lambda i: scores[i], reverse=True)
    rank_of = {layer: rank for rank, layer in enumerate(order)}
    for row in rows:
        marker = ""
        if keep is not None:
            marker = "  keep" if row["layer"] in keep else "  drop"
        print(
            f"{row['layer']:>5}  {row['relative_delta']:>10.5f}  {row['cosine']:>8.5f}   "
            f"{rank_of[row['layer']]:>3}{marker}"
        )
    if keep is not None:
        print(f"\n--student-layer-indices {','.join(str(i) for i in keep)}")
    print(f"\nWrote {out_dir / 'block_importance.json'}")


def apply_no_network(args: argparse.Namespace) -> None:
    """Mirror the offline setup the other cache_head entry points use."""
    if not args.no_network:
        return
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.model_base_path
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.setdefault(key, "1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rank Wan DiT blocks by residual-stream contribution, to choose "
                    "which blocks a shallower sparse student should keep."
    )
    parser.add_argument("--captions", required=True, help="MixKit caption JSONL path")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--model-base-path", default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=15)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--sigma-shift", type=float, default=5.0)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--keep", type=int, default=None,
        help="suggest this many blocks to retain (printed as --student-layer-indices)",
    )
    parser.add_argument("--out-dir", default="block_importance")
    parser.add_argument(
        "--negative-prompt",
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
                "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
                "杂乱的背景，三条腿，背景人很多，倒着走",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.keep is not None and args.keep < 1:
        raise SystemExit("--keep must be positive")
    apply_no_network(args)
    run_profile(args)


if __name__ == "__main__":
    sys.exit(main())
