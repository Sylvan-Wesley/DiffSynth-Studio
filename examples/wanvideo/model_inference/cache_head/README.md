# CacheHead: DMD-based Cache Distribution Matching for Wan2.1

A 15-step Wan generator that keeps vanilla Wan frozen and replaces eight
expensive denoising calls with a lightweight residual **CacheHead**.  The
default 1-indexed full-Wan anchor steps are `[1, 2, 3, 4, 5, 6, 7]`; steps
`[8, 9, 10, 11, 12, 13, 14, 15]` use CacheHead.  The schedule is
configuration-only, so later experiments can move anchors without changing
model code.

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
| `sparse_attention.py` | Pluggable sparse self-attention: drop-in replacement for Wan's parameter-free `AttentionModule`, plus the pattern registry and block-mask cache |
| `sparse_cache_head.py` | `SparseCacheHead` — the teacher-inherited DiT student — and its conv3d latent-fusion adapter, depth selection, and checkpoint I/O |
| `profile_block_importance.py` | Ranks teacher DiT blocks by residual-stream contribution, to choose which blocks a shallower student keeps |
| `cache_head_error_heatmap.py` | Per-patch head-vs-teacher error heat maps over a full trajectory (panel grid + per-step summary curve), plus the decoded video of that same rollout |
| `pca_trajectory_eval.py` | Shared-PCA trajectory-difference artifacts (npz / png / metrics json) |
| `cache_head_harness.py` | Agent harness loop: tamper-evident ledger, locked manifest, 7-invariant verify, runner, evaluate, review |
| `tests/` | CPU-runnable tests (no GPU or Wan weights needed) |

## The `sparse_dit` head variant

`--head-variant sparse_dit` replaces the lightweight token-space head with a
*structural clone of the teacher*: the same Wan DiT, with every self-attention
swapped for a sparse one, fed by a conv3d that fuses the previous guided
velocity into the current latent.

```
prev_guided [B,S,64] --unpatchify--> [B,16,f,h,w] --.
                                                    |--concat--> conv3d --+--> DiT --> v_hat [B,S,64]
current latent x_k   ----------------------------- '                      |
                                                     current latent ------'  (residual)
```

- **Weights are inherited**, not retrained from scratch. Wan's `AttentionModule`
  owns no parameters, so swapping it leaves `state_dict()` byte-identical.
- **The conv's last layer is zero-initialized**, so at step 0 the fused input is
  exactly `x_k`, the DiT sees precisely its native input distribution, and no
  warm-up phase is needed. With `--sparse-pattern dense` the student then
  reproduces a positive-context teacher forward *bit-for-bit* (pinned by
  `test_dense_student_reproduces_the_teacher_exactly`).
- **A head step is a single forward with positive context only.** The student
  predicts the CFG-guided velocity outright; classifier-free guidance is
  distilled into its weights. There is no residual add onto `prev_guided` —
  the previous prediction reaches the model only through the conv fusion.
  Early loss is therefore large by construction; `carry_mse` and
  `relative_improvement` are the meaningful references, not the absolute value.
- **Everything trains**: the full DiT plus the adapter (~1.3B parameters,
  roughly 10 GB/rank of params + grads + AdamW state in bf16). Activation
  checkpointing is on by default and is effectively required at S=32,760.

### Cost, and why depth matters

Per block at S=32,760 (grid 21x30x52, dim 1536, ffn 8960):

| component | MACs | share |
|---|---|---|
| self-attn scores | 3.30e12 | 70% |
| self-attn projections | 0.31e12 | 7% |
| cross-attn | 0.20e12 | 4% |
| FFN | 0.90e12 | 19% |

A 5x5 spatial window over 5 frames is 125 of 32,760 keys (0.38% density),
collapsing the 70% term to almost nothing — after which the **FFN dominates at
~63%** and depth becomes the main remaining lever. Hence `--student-num-layers`
(uniform stride retaining the first and last block) and
`--student-layer-indices` (explicit, chosen from `profile_block_importance.py`).

```bash
# 1. measure which blocks actually move the residual stream
python profile_block_importance.py --captions mixkit_captions.jsonl --keep 15

# 2. sanity check: dense + full depth must match the teacher
python cache_head_model_training.py --arm supervised --head-variant sparse_dit \
    --sparse-pattern dense --captions mixkit_captions.jsonl --subset 2 --epochs 1 \
    --micro-batch 1 --batch-size 1

# 3. the real run
python cache_head_model_training.py --arm supervised --head-variant sparse_dit \
    --sparse-pattern spatiotemporal_window --sparse-spatial-radius 2 \
    --sparse-temporal-radius 2 --student-layer-indices 0,2,4,... \
    --captions mixkit_captions.jsonl --prefetch-and-offload
```

`--prefetch-and-offload` warms the teacher-trajectory and prompt-embedding
caches for the whole split, then moves the teacher DiT (~2.6 GB) and umT5-XXL
(~11 GB) to CPU — neither is needed once both caches are warm, and that memory
is better spent on the student's activations.

Notes and caveats:

- Prompt embeddings get their own cache (`<trajectory-dir>/contexts/`) keyed by
  caption alone, since they do not depend on the schedule, seed, or cfg scale.
- `sparse_dit` currently requires `--arm supervised`; the DMD arms roll the
  student forward through `head_step` and have not been validated at full size.
- Checkpoints grow from ~300 KB to ~2.6 GB, so pick `--checkpoint-every`
  accordingly.
- Set `CACHEHEAD_COMPILE_FLEX=0` to run `flex_attention` eagerly while
  debugging; it is compiled by default because eager flex_attention
  materializes the score matrix.
- Training is teacher-forced on the *teacher's* `x_k`, while inference sees the
  drifted live hybrid latent — the same train/test gap the latent-conditioned
  variants have.

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
  5 fake-score updates.  Only CacheHead weights + config are exported.
- The **`supervised`** arm runs a different loop from the DMD arms (the loop is
  derived from `--arm`, not a separate switch):
  - A full frozen-Wan teacher supplies every scheduler update.  Its 15 guided
    CFG token predictions are cached persistently, and student step `k` uses
    teacher step `k-1` as input and teacher step `k` as its velocity target.
  - Supervision is plain float32 MSE in noise-token space `[B, N, C]`, averaged
    over the schedule complement (steps 8--15 by default).  The student never
    changes the training trajectory.
  - Each data iteration performs `--optimizer-steps-per-iteration` fresh
    optimizer updates over the same teacher batch (default 5).
  - Prompts are batched (`--micro-batch`, which must divide `--batch-size`); the
    startup probe reports the peak for that micro-batch and fails fast rather
    than searching, so the effective batch is reproducible across runs and ranks.
  - Epoch loop over the train split with per-epoch validation on the held-out
    `val` split, per-head-step val loss, and a best-val checkpoint
    (`cache_head_best.ckpt`).
  - Missing trajectories are generated lazily and atomically under
    `--trajectory-dir`; valid files are loaded in later epochs or runs.
  - This arm skips the LoRA fake-score clone entirely, freeing ~2.6 GB (bf16)
    for a larger micro-batch.
- `--no-network` runs fully offline: it sets `DIFFSYNTH_SKIP_DOWNLOAD=true` (so
  `ModelConfig.download_if_necessary` resolves `<base>/<model_id>/<pattern>` by
  glob instead of calling modelscope), passes `skip_download=True` on every
  `ModelConfig`, sets `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` (the umt5
  tokenizer loads through `AutoTokenizer.from_pretrained` and would otherwise
  revision-check), and turns W&B off entirely.
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
    --epochs 20 --batch-size 8 --micro-batch 4 \
    --optimizer-steps-per-iteration 5 \
    --trajectory-dir runs/supervised/trajectories \
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

# Standalone error heat map from a checkpoint (GPU).  Writes the panel grid,
# the raw error arrays, and the decoded video of the same rollout, so the
# picture and the footage always describe one run.
python cache_head_error_heatmap.py --checkpoint runs/supervised/cache_head_best.ckpt \
    --captions mixkit_captions.jsonl --num-prompts 2 --out-dir heatmaps
#   heatmaps/cache_head_error_heatmap.png
#   heatmaps/cache_head_error_heatmap.pt
#   heatmaps/rollout-0.mp4, rollout-1.mp4      (--video PATH to rename, --no-video to skip)

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
