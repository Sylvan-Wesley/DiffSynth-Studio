"""Why do two CacheHead checkpoints produce identical video?

Answers, in order of how often it is the cause:

  1. The head never left zero-init, so both checkpoints are literally
     ``carry_previous`` and every hybrid run is byte-identical.  On the ``dmd``
     arm with ``--warmup-steps 0`` this is the expected early behaviour: the
     LoRA fake-score is zero-delta at init, so ``fake_x0 == teacher_x0``, the
     DMD loss is exactly 0, and the head receives exactly no gradient until the
     fake-score has learned to differ from the teacher.

  2. The weights differ, but the residual is small enough that
     ``prev_guided + residual`` rounds back to ``prev_guided`` in bf16
     (~8 mantissa bits: a residual below ~0.2% of the token magnitude is
     mostly swallowed).

  3. The checkpoint files are the same file.

Usage:
    python diagnose_checkpoints.py runs/vanilla_dmd/cache_head_100.ckpt \\
                                   runs/vanilla_dmd/cache_head_200.ckpt
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from cache_head_model import load_cache_head


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="two or more CacheHead checkpoints")
    ap.add_argument("--tokens", type=int, default=4096, help="synthetic token count")
    ap.add_argument("--timestep", type=float, default=500.0)
    args = ap.parse_args()

    paths = [Path(p) for p in args.checkpoints]
    heads = []
    print("=" * 72)
    for p in paths:
        head, cfg = load_cache_head(p)
        dtype = next(head.parameters()).dtype
        out_norm = head.out_proj.weight.float().norm().item()
        print(f"{p.name:32s} sha={_sha(p)}  dtype={dtype}")
        print(f"{'':32s} |out_proj| = {out_norm:.6e}"
              + ("   <-- STILL ZERO-INIT (== carry_previous)" if out_norm == 0.0 else ""))
        heads.append((p, head, cfg))

    # --- do the weights actually differ? ---
    print("=" * 72)
    base_name, base_head, _ = heads[0]
    for name, head, _ in heads[1:]:
        deltas = []
        for (k, a), (_, b) in zip(base_head.state_dict().items(), head.state_dict().items()):
            d = (a.float() - b.float()).abs().max().item()
            if d > 0:
                deltas.append((k, d))
        if not deltas:
            print(f"{base_name.name} vs {name.name}: WEIGHTS ARE BIT-IDENTICAL")
        else:
            print(f"{base_name.name} vs {name.name}: {len(deltas)} tensors differ, "
                  f"max delta {max(d for _, d in deltas):.3e}")
            for k, d in sorted(deltas, key=lambda kv: -kv[1])[:5]:
                print(f"    {k:40s} {d:.3e}")

    # --- is the residual big enough to survive bf16? ---
    print("=" * 72)
    print("Residual magnitude vs a unit-scale token grid, and whether the")
    print("bf16 add `prev_guided + residual` actually changes anything:")
    grid = (4, 8, 8)
    n_tok = grid[0] * grid[1] * grid[2]
    torch.manual_seed(0)
    for name, head, cfg in heads:
        dtype = next(head.parameters()).dtype
        tokens = torch.randn(1, n_tok, cfg.token_channels, dtype=dtype)
        t = torch.tensor([args.timestep], dtype=dtype)
        with torch.no_grad():
            residual = head(tokens, t, grid)
            merged = tokens + residual
        ratio = (residual.float().abs().mean() / tokens.float().abs().mean()).item()
        unchanged = (merged == tokens).float().mean().item()
        verdict = ("NO-OP: hybrid output will equal carry_previous"
                   if unchanged == 1.0 else
                   f"{(1 - unchanged) * 100:.1f}% of tokens change")
        print(f"  {name.name:32s} |res|/|tok| = {ratio:.3e}  -> {verdict}")
    print("=" * 72)


if __name__ == "__main__":
    main()
