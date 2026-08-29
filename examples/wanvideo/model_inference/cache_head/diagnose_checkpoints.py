"""Why do two CacheHead checkpoints produce identical video?

Answers, in order of how often it is the cause:

  1. The head's residual is effectively zero, so both checkpoints are
     ``carry_previous`` and every hybrid run is byte-identical.  On the ``dmd``
     arm with ``--warmup-steps 0`` this is the expected early behaviour: the
     LoRA fake-score is zero-delta at init, so ``fake_x0 == teacher_x0``, the
     DMD loss is exactly 0, and the head receives exactly no gradient until the
     fake-score has learned to differ from the teacher.

  2. The weights differ, but the residual is small enough that
     ``prev_guided + residual`` rounds back to ``prev_guided`` in bf16
     (~8 mantissa bits: a residual below ~0.2% of the token magnitude is
     mostly swallowed).  Judge this at the *real* token scale -- CacheHead is
     RMSNorm-fronted, so its output magnitude is independent of its input, and
     a residual that looks healthy against unit-scale tokens can vanish
     entirely against the real guided tokens.

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

    # --- is the residual big enough to survive the add, at real token scale? ---
    print("=" * 72)
    print("CacheHead is RMSNorm-fronted, so |residual| does NOT grow with")
    print("|prev_guided|.  The absolute figure is the head's true output scale;")
    print("whether it survives `prev_guided + residual` depends on how large the")
    print("real Wan guided tokens are.  Run inference once and read the")
    print("'[head effect] mean |tokens|=' line to find your actual scale.")
    print()
    grid = (4, 8, 8)
    n_tok = grid[0] * grid[1] * grid[2]
    scales = [1.0, 5.0, 20.0, 50.0, 200.0]
    header = "".join(f"{s:>10.0f}" for s in scales)
    print(f"  {'checkpoint':32s} {'|residual|':>11s}   % tokens changed by a bf16 add at |tok| =")
    print(f"  {'':32s} {'':>11s}   {header}")
    torch.manual_seed(0)
    for name, head, cfg in heads:
        head32 = head.float()
        base = torch.randn(1, n_tok, cfg.token_channels)
        t = torch.tensor([args.timestep])
        with torch.no_grad():
            residual = head32(base, t, grid)
        row = ""
        for scale in scales:
            tok_bf = (base * scale).to(torch.bfloat16)
            res_bf = residual.to(torch.bfloat16)
            changed = (tok_bf + res_bf != tok_bf).float().mean().item()
            row += f"{changed * 100:9.1f}%"
        print(f"  {name.name:32s} {residual.abs().mean().item():11.3e}   {row}")
    print()
    print("A row that collapses to ~0% at your real token scale means the head is")
    print("a no-op in bf16 and hybrid output will match carry_previous exactly --")
    print("regardless of whether the checkpoint loaded correctly.")
    print("=" * 72)


if __name__ == "__main__":
    main()
