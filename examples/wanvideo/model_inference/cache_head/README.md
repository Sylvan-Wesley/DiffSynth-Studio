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
| `download_mixkit_captions.py` | Downloads the upstream Open-Sora-Plan annotation JSON and extracts its 8,230 current MixKit captions into training JSONL |
| `cache_head_model_inference.py` | Hybrid inference runner (`hybrid` / `full` / `carry` modes), 16-state trajectory capture |
| `cache_head_model_training.py` | Training harness + loss study: `carry_previous`, `residual_regression`, `supervised`, `dmd`, `dmd_plus_reg` |
| `cache_head_error_heatmap.py` | Per-patch head-vs-teacher error heat maps over a full trajectory (panel grid + per-step summary curve) |
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
- The **`supervised`** arm runs a different loop from the DMD arms (the loop is
  derived from `--arm`, not a separate switch):
  - One batched hybrid rollout supervises **every** head step against frozen Wan
    queried at that same hybrid state, so a rollout yields `10 × B` targets
    instead of one.  The loss lives in noise-token space `[B, N, C]`; the
    unpatchified latent only advances the scheduler.
  - Prompts are batched (`--micro-batch`, which must divide `--batch-size`); the
    startup probe reports the peak for that micro-batch and fails fast rather
    than searching, so the effective batch is reproducible across runs and ranks.
  - Epoch loop over the train split with per-epoch validation on the held-out
    `val` split, per-head-step val loss, and a best-val checkpoint
    (`cache_head_best.ckpt`).
  - The prefix is the **hybrid student rollout** (the head drives prior head
    steps), so states match deployment.  `--chain-run-grads` additionally
    backpropagates through consecutive head-step runs.
  - This arm skips the LoRA fake-score clone entirely, freeing ~2.6 GB (bf16)
    for a larger micro-batch.
- `--no-network` runs fully offline: it sets `DIFFSYNTH_SKIP_DOWNLOAD=true` (so
  `ModelConfig.download_if_necessary` resolves `<base>/<model_id>/<pattern>` by
  glob instead of calling modelscope), passes `skip_download=True` on every
  `ModelConfig`, sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` (the umt5
  tokenizer loads through `AutoTokenizer.from_pretrained` and would otherwise
  revision-check), and turns W&B off unless `--wandb-mode offline` is given.
  Point `--model-base-path` at the directory *containing* `<model-id>/`.
- `--heatmap-every N` renders a per-patch error heat map every N epochs:
  `‖v_head − v_teacher‖₂` over the 64 token channels, reshaped to the `(f, h, w)`
  token grid, on one shared color scale across all 15 steps.  Read it against the
  per-head-step val loss: the loss says *when* the head drifts, the map says *where*.

## Running

Requires a GPU box with the Wan2.1-T2V-1.3B weights and a MixKit caption corpus
as a JSONL (`{"id": ..., "caption": ...}`).  `--captions <path>` points
the training harness at it.

Download the captions without downloading the 27 GB MixKit video archive:

```bash
python download_mixkit_captions.py --output mixkit_captions.jsonl
```

The downloader reads the publicly released Open-Sora-Plan v1.0 ShareGPT4V
annotation JSON, filters MixKit records, verifies the currently published
8,230 captions, and writes the exact JSONL contract above. The earlier 6,484
figure referred to a curated subset; the training harness does not require it.
It downloads approximately
474 MB of annotations into the standard Hugging Face cache, so subsequent runs
reuse it without downloading again.  Pass `--cache-dir <path>` to place that
cache on a volume with more space.

```bash
# CPU tests (any machine with torch + einops)
python -m pytest tests/ -q

# Hybrid inference (GPU)
python cache_head_model_inference.py --checkpoint cache_head_final.ckpt \
    --prompt "..." --output out.mp4 --trajectory traj.npz

# Vanilla DMD training on eight GPUs (no regression loss or warm-up)
pip install -e '.[wandb]'  # omit this line and --wandb-project to run without W&B
torchrun --standalone --nproc_per_node=8 cache_head_model_training.py \
    --arm dmd --warmup-steps 0 --updates 10000 \
    --captions mixkit_captions.jsonl --batch-size 8 --precision bf16 \
    --save-dir runs/vanilla_dmd \
    --wandb-project cache-head-dmd --wandb-run-name dmd-8xa100-run1

# W&B records this as: dmd-8xa100-run1-YYYYMMDD-HHMMSS+ZZZZ

# Supervised training on eight GPUs (batched rollouts, epoch + val loop)
torchrun --standalone --nproc_per_node=8 cache_head_model_training.py \
    --arm supervised --captions mixkit_captions.jsonl \
    --epochs 20 --batch-size 8 --micro-batch 4 --reg-loss huber \
    --val-subset 128 --heatmap-every 5 --precision bf16 \
    --save-dir runs/supervised \
    --wandb-project cache-head-supervised

# Find the largest micro-batch first: raise it until the startup probe OOMs,
# then step back one.  --batch-size must stay a multiple of --micro-batch.

# Offline / air-gapped box: no modelscope, no HuggingFace, no W&B.
# Weights must already sit at <model-base-path>/<model-id>/, e.g.
#   /data/wan_models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model*.safetensors
#   /data/wan_models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth
#   /data/wan_models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth
#   /data/wan_models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
torchrun --standalone --nproc_per_node=8 cache_head_model_training.py \
    --arm supervised --captions mixkit_captions.jsonl \
    --no-network --model-base-path /data/wan_models \
    --epochs 20 --batch-size 8 --micro-batch 4 --save-dir runs/supervised

# Standalone error heat map from a checkpoint (GPU)
python cache_head_error_heatmap.py --checkpoint runs/supervised/cache_head_best.ckpt \
    --captions mixkit_captions.jsonl --num-prompts 2 --out-dir heatmaps

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
