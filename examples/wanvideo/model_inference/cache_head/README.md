# CacheHead: DMD-based Cache Distribution Matching for Wan2.1

A 15-step Wan generator that keeps vanilla Wan frozen and replaces ten
expensive denoising calls with a lightweight residual **CacheHead**.  The
default 1-indexed full-Wan anchor steps are `[1, 2, 6, 10, 14]`; the other ten
steps use CacheHead.  The schedule is configuration-only, so later experiments
can move anchors without changing model code.

At a head step:

```
v_hat_i = v_{i-1} + r_phi(v_{i-1}, t_i)
```

`v_{i-1}` is exactly the nearest preceding guided noise-token prediction
(`[B, S, 64]`).  The deployed head sees only those tokens and the current Wan
timestep.  Wan, scheduler, and CFG remain unchanged.  No RAS, sparse KV,
MotionCache, token selection, or selector training is used.

## Layout

| File | Role |
|---|---|
| `cache_head_model.py` | CacheHead network (RMSNorm → timestep AdaLN → channel MLP → depthwise 3D token-grid mixer → timestep AdaLN → zero-init out-proj), schedule + config dataclasses, checkpoint I/O |
| `fake_score_wan.py` | Strict-DMD fake-score estimator: a LoRA Wan (frozen Wan DiT clone + trainable low-rank adapters). Training-only; never exported |
| `download_mixkit_captions.py` | Downloads the upstream Open-Sora-Plan annotation JSON and extracts the 6,484 MixKit captions into training JSONL |
| `cache_head_model_inference.py` | Hybrid inference runner (`hybrid` / `full` / `carry` modes), 16-state trajectory capture |
| `cache_head_model_training.py` | Training harness + loss study: `carry_previous`, `residual_regression`, `dmd`, `dmd_plus_reg` |
| `pca_trajectory_eval.py` | Shared-PCA trajectory-difference artifacts (npz / png / metrics json) |
| `cache_head_harness.py` | Agent harness loop: tamper-evident ledger, locked manifest, 7-invariant verify, runner, evaluate, review |
| `tests/` | CPU-runnable tests (51 tests; no GPU or Wan weights needed) |

## Key design facts

- Noise tokens are `[B, S, 64]` (Wan head output: `C_tok = out_dim·∏patch = 16·(1·2·2)`),
  token grid `S = f·h·w` with `f = latent_frames`, `h = latent_h/2`, `w = latent_w/2`,
  row-major with `f` slowest.  `unpatchify_tokens` mirrors Wan's `unpatchify`.
- The output projection is **zero-initialized**, so a fresh head emits exactly zero
  residual and reproduces `carry_previous` bit-for-bit.
- DMD loss follows the repo convention (`diffsynth/diffusion/dmd2.py`):
  `flow_to_x0 = latents − σ·flow`, weight `w = 1/(|x0 − teacher_x0|.mean + 1e-6)`,
  `L = 0.5·‖x0 − sg[x0 − w·(fake_x0 − teacher_x0)]‖²`.
- Training arms: 2,000 shared regression warm-up, then 10,000 CacheHead updates
  per arm; bf16, AdamW, effective batch 8, gradient clipping 1.0, startup memory
  probe for micro-batch / accumulation.  DMD alternates 1 CacheHead update with
  4 fake-score updates.  Only CacheHead weights + config are exported.

## Running

Requires a GPU box with the Wan2.1-T2V-1.3B weights and the MixKit 6,484-caption
corpus as a JSONL (`{"id": ..., "caption": ...}`).  `--captions <path>` points
the training harness at it.

Download the captions without downloading the 27 GB MixKit video archive:

```bash
python download_mixkit_captions.py --output mixkit_captions.jsonl
```

The downloader reads the publicly released Open-Sora-Plan v1.0 ShareGPT4V
annotation JSON, filters MixKit records, verifies that 6,484 captions were
found, and writes the exact JSONL contract above.  It downloads approximately
474 MB of annotations; pass `--keep-source` to retain that source JSON.

```bash
# CPU tests (any machine with torch + einops)
python -m pytest tests/ -q

# Hybrid inference (GPU)
python cache_head_model_inference.py --checkpoint cache_head_final.ckpt \
    --prompt "..." --output out.mp4 --trajectory traj.npz

# Train one arm (GPU)
python cache_head_model_training.py --arm dmd_plus_reg --reg-weight 0.1 \
    --captions mixkit_captions.jsonl --save-dir out

# PCA trajectory-difference evaluation (GPU)
python pca_trajectory_eval.py --checkpoint out/cache_head_final.ckpt \
    --prompts-jsonl heldout_test.jsonl --panel-size 8 --seeds 0,1,2 --out-dir pca

# Agent harness loop (ledger / manifest / verify / run / review)
python cache_head_harness.py ledger-new --ledger ledger.jsonl
python cache_head_harness.py manifest --hypothesis dmd \
    --prompts-jsonl heldout_test.jsonl --out-dir run1
python cache_head_harness.py verify --out-dir run1
python cache_head_harness.py run --out-dir run1
python cache_head_harness.py review --ledger ledger.jsonl
```

## Promotions

A candidate is promoted only when it (all three seeds): has no NaN/Inf,
schedule violations, or late-step trajectory explosion; improves held-out
hybrid-vs-full velocity error by ≥10% over carry_previous; does not regress
alignment or temporal metrics vs carry_previous; achieves ≥1.5× denoising
speedup over 15 full-Wan calls; and produces complete finite PCA artifacts for
the locked probe panel.  PCA coordinates are visual diagnostics only — go/no-go
uses the full-dimensional latent metrics.
