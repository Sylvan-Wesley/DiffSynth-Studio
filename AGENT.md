# AGENT.md — RAS Implementation (DiffSynth-Studio)

Training-free inference acceleration for **Wan2.1-T2V-1.3B** via Region-Adaptive Sampling
(Liu et al., "Region-Adaptive Sampling for Diffusion Transformers", CVPR 2026). Each denoising
step runs only a fraction (`ratio`) of video-latent tokens through the full DiT stack; the rest
get a cheap "dumb" update. Goal: ~2x speedup with minimal quality loss.

All RAS work lives on the local `ras` branch. **`origin/main` is the upstream modelscope repo
and contains NO RAS code** — never base RAS changes on it.

## Files
- `examples/wanvideo/model_inference/RAS-Wan2.1-T2V-1.3B.py` — main inference script. Owns all
  RAS state (KV caches, skip counters, dense warm-up, per-condition noise caches), runs CFG,
  decodes. Config knobs at top (`ratio`, `num_dense_steps`, `dumb_update`, `enable_viz`,
  `viz_mask_mode`).
- `examples/wanvideo/model_inference/benchmark_RAS_vs_Wan.py` — times full inference
  (`kv_cache=None`) vs RAS (`kv_cache`+`ratio`) through `WanModel.forward()` directly.
- `examples/wanvideo/model_inference/visualize_ras_masks.py` — heatmap overlay of per-step
  selection masks.
- `diffsynth/models/wan_video_dit.py` — ALL RAS machinery (module helpers + `WanModel` methods).

## Structure of `diffsynth/models/wan_video_dit.py` (grep-verified line refs)
Module helpers:
- `flash_attention` (30) — FA3/FA2/Sage dispatch.
- `gather_tokens` (66) / `scatter_tokens_` (75) — gather/scatter token rows by `[B,N]` indices.
- `update_noise_cache` (88) — scatter the active tokens' prediction into a retained full-sequence
  noise cache; used for the posi, nega, and guided caches (see "Per-step data flow", step 7).
- `cache_ready` (108) — is the KV cache populated?
- `selection_mask_to_grid` (115) — `[B,S]` bool mask → `[B,1,f,h,w]` grid for viz.
- `rope_apply` (160) — RoPE, real-valued bf16 (see Gotchas).
- `set_to_torch_norm` (205) — switches RMSNorm to fused `F.rms_norm`.

Model:
- `SelfAttention.forward` (260) — with cache + `selected_patches`: compute q for active tokens;
  compute k/v for active and **scatter back** into full cached tensors; inactive read stale k/v.
- `CrossAttention.forward` (312) — ctx K/V cached per layer; q computed only for active tokens.
- `DiTBlock.forward` (380) — threads `selected_patches` + both caches through self/cross/ffn.
- `Head.forward` (428) — outputs token-level noise `[B, S, out_dim*prod(patch_size)]`.
- `WanModel` (489) — state: `_last_selected_patches` (574), `debug_masks` (577); `patchify` (651);
  `unpatchify` (667); `select_region` (674); `update_skip_record` (722); `get_selection_masks`
  (738); `clear_selection_masks` (745); `get_last_selected_patches` (749); `forward` (757).
  NO internal noise cache — previous predictions are passed in: `prev_noise_tokens` feeds
  `select_region` (the guided combo) and `dumb_noise_tokens` feeds this branch's dumb fill
  (posi→prev-posi, nega→prev-nega; falls back to `prev_noise_tokens`); the full token-space
  prediction is returned via `return_noise_tokens`.

## Per-step data flow
1. **Dense warm-up** (`num_dense_steps`, default 20) runs ALL tokens — seeds per-layer KV caches
   and initializes the per-condition noise caches (`update_noise_cache` returns the full prediction
   when there is no prior). The script also forces a few mid-run dense steps (e.g. indices 30/40)
   to refresh stale KV caches. Sparse steps follow.
2. **Sparse step** — `select_region` (positive branch only) picks top-k tokens:
   `importance = 1/(std(prev_noise)+eps)` (low variance = semantically settled = safe to skip),
   `score = importance * exp(k_starvation * skip_k)` (starvation prevention). `prev_noise_tokens`
   is the PREVIOUS step's CFG-combined prediction, so selection follows the scheduler's actual
   trajectory. First step has no prev noise → falls back to L2 norm of latents.
3. `update_skip_record` — reset selected tokens' drop counter to 0, then increment all
   (unselected accumulate, selected → 1). Called ONLY on the positive branch; the negative branch
   reuses the selection (step 4) so the record isn't double-updated. Dense steps instead zero
   `skip_k` inside `forward()` — a dense pass refreshes every token, so none should carry a stale
   starvation boost into the next sparse selection.
4. **Selection sharing** — after every forward, `_last_selected_patches` holds the indices used
   and `get_last_selected_patches()` returns them. The script passes them to the NEGATIVE branch
   as `selected_patches`, so both CFG branches process the same tokens. This avoids re-deriving
   the selection from an independently chosen noise map and double-updating the skip record.
5. DiT blocks run ONLY on gathered active tokens (see attention notes above).
6. **Head / dumb update** — active: fresh prediction. Inactive: strategy from `dumb_update`
   (default `"Previous"`):
   - `"Previous"` — carry forward the cached value for that token from THIS branch's own condition
     cache (`dumb_noise_tokens`: posi→prev-posi, nega→prev-nega).
   - `"Zero"` — predict zero noise for inactive tokens.
   - any other value → `ValueError`.
   head-on-raw is a fallback ONLY when `dumb_noise_tokens is None` (head is trained on DiT-block
   output, so raw input gives garbage). `forward()` returns the full token-space noise as
   `noise_tokens` when `return_noise_tokens=True`.
7. **Per-condition cache update** (script side) — after CFG the script scatters ONLY the active
   positions into three retained caches via `update_noise_cache` (88): `noise_posi_tokens` →
   `prev_posi_noise_tokens`, `noise_nega_tokens` → `prev_nega_noise_tokens`, and the guided combo
   `nega + cfg*(posi − nega)` → `prev_guided_noise_tokens`. Because `unpatchify` is a pure
   `rearrange` (linear), token-space guided noise == the latent-space `noise_pred` the scheduler
   consumes, so the caches always match the denoising trajectory. Keeping posi/nega separate makes
   the CFG formula a genuine `nega + cfg*(posi − nega)` on EVERY token: each branch's inactive
   entries are real estimates of its own condition, not the guided value (a single shared cache
   filled both branches with the guided value and only worked via a fragile collapse). The caches
   update ONCE per step, never inside a branch. With `dumb_update="Zero"`, inactive tokens are
   zeroed for the scheduler but the caches retain their prior real values — otherwise the next
   selection would treat them as infinitely important (std = 0).
8. Unpatchify → `noise_pred` → `scheduler.step`.

## Gotchas (don't rediscover these)
- **Token count S is AFTER patchify**: `S = f * (h//patch_size[1]) * (w//patch_size[2])`, NOT VAE
  latent dims. Wrong S trips the bounds check in `forward` → IndexError.
- **OOM (79GB GPU) history — preserve these fixes**:
  - Denoising loop MUST run under `torch.inference_mode()`; autograd otherwise saves ~3GB/block ×
    30 blocks.
  - `rope_apply` must stay real-valued bf16 — no float32/float64 temporaries.
  - RMSNorm: `set_to_torch_norm` (fused `F.rms_norm`) avoids full float32 copies.
  - `expandable_segments` was REMOVED — it blocked T5 GPU memory release.
  - Offload T5 text encoder + VAE to CPU after prompt encoding / before decode.
- `prev_posi_noise_tokens` / `prev_nega_noise_tokens` / `prev_guided_noise_tokens` are SCRIPT-owned
  state (not model attributes) reset to `None` before the loop; the model's `_last_selected_patches`
  is reset in `__init__`. Sparse steps with no prior prediction hit the head-on-raw fallback (keep
  it working).
- The NEGATIVE CFG branch MUST receive the positive branch's `selected_patches` (via
  `get_last_selected_patches()`) on sparse steps — never let it call `select_region` again.
- The noise caches are RAS state, not per-branch state: update each one ONCE per step AFTER CFG.
  Never store a single branch's raw prediction — the old internal `_prev_noise_tokens` was
  last-writer-wins, so the cache ended up holding the nega branch's prediction and selection
  stopped tracking the guided trajectory.
- Feed each CFG branch its OWN condition cache as `dumb_noise_tokens` (posi→prev-posi,
  nega→prev-nega); pass the guided combo as `prev_noise_tokens` only where selection happens (the
  positive branch). If a branch carries the wrong condition's cache, the inactive-token CFG
  combination silently becomes `nega + cfg·(posi − nega)` over mismatched conditions.
- All scripts pass `dumb_update` to BOTH CFG branches (it used to be positive-only; nega silently
  defaulted to `"Previous"`). Keep the two branches on the same mode so the inactive-token values
  stay consistent.
- KV caches persist across steps → dense warm-up required before sparse steps.
- Eval-only; `inference_mode` forbids autograd.

## Running
- Model: `Wan-AI/Wan2.1-T2V-1.3B`. Defaults: `ratio=0.25`, `num_dense_steps=20`,
  `dumb_update="Previous"`, `num_inference_steps=50`, `cfg_scale=5.0`, 81 frames @ 480×832.
- `python examples/wanvideo/model_inference/RAS-Wan2.1-T2V-1.3B.py` → `ras_output.mp4`; with
  `enable_viz=True`, per-step selection masks under `ras_masks/` (format via `viz_mask_mode`).

---

## Flow-guided RAS experiment (branch `ras-flow-bootstrap`)

Training-free acceleration whose sparse selection is driven by **static optical-flow magnitude**
instead of the previous-noise variance metric. Flow is estimated ONCE from a fully generated dense
reference video and held fixed during the RAS pass; only the starvation counts evolve.

### Files
- `examples/wanvideo/model_inference/RAS-Wan2.1-T2V-1.3B-OpticalFlow.py` — the two-pass experiment.
  Module-level pure helpers (`spatial_pool_to_patch_grid`, `temporal_group_flow`, `flow_grid_to_b_s`,
  `estimate_flow_magnitudes`) are importable for unit tests; all model work lives in `main()` under
  `if __name__ == "__main__":`. The existing `RAS-Wan2.1-T2V-1.3B.py` is untouched — it stays the
  non-flow baseline.
- `diffsynth/models/wan_video_dit.py` — same file as the base RAS machinery; adds OPTIONAL flow-ranking
  inputs (see below). RAFT is never a model-layer dependency.
- `tests/test_ras_flow_bootstrap.py` — CPU unit tests (plain-assert runner, no pytest).

### Two-pass lifecycle
1. **Initial latents** — `pipe.generate_noise(shape, seed, rand_device="cpu")` ONCE; the dense pass and
   the RAS pass each clone it (`initial_latents.clone()`).
2. **Pass 1 — dense reference** — ordinary full-Wan CFG denoising for ALL scheduler steps with
   `kv_cache=None` (`noise_pred = nega + cfg*(posi - nega)`), decoded and saved to
   `wan_flow_reference.mp4`. This video is the flow source AND the visual baseline.
3. **Flow** — RAFT-Small (torchvision) over every adjacent decoded RGB frame pair → per-pixel magnitude
   `sqrt(u²+v²)` → spatial avg-pool to the DiT patch grid → temporal aggregation to the latent-frame
   grid (see mapping below) → flatten to `[B, S]` frame-major (same order as `WanModel.patchify`).
4. **Pass 2 — flow-guided RAS** — fresh KV/noise caches, skip counters, debug masks; `latents =
   initial_latents.clone()`; dense warm-up steps seed the KV caches, then every sparse step passes
   `flow_magnitudes` + `starvation_scale` to `WanModel.forward`. Decoded to `ras_flow_ranked.mp4`.

### Score definition (flow-guided sparse steps)
```
score_i = max(flow_magnitude_i, 1e-6) * exp(starvation_scale * starvation_count_i)
```
selects the `ceil(ratio * S)` highest-scoring tokens. The `1e-6` magnitude floor keeps the
multiplicative starvation term effective for zero-motion patches — without it a static region would
score 0 forever and starve regardless of skip count.

### Latent/RGB temporal mapping (causal Wan decoder)
Wan's VAE decoder is causal with `T_RGB = 4*T_latent - 3`. Flow pairs map to latent frames as:
- latent frame 0 ← flow pair 0;
- latent frame k ≥ 1 ← flow pairs `[4k-3, 4k]` (four pairs each);
- the final partial group is averaged normally (mean over the remaining pairs).

Spatial pooling uses `F.adaptive_avg_pool2d` to `(latent_h//patch_size[1], latent_w//patch_size[2])`, so
it is robust to the exact `H/W ÷ 16` divisibility. RAFT inputs must have H,W divisible by 8.

### Model hook — `wan_video_dit.py`
- `validate_flow_magnitudes(flow, x)` (module helper): enforces `[B, S]` (S = token count AFTER
  patchify), moves onto `x.device`, requires finite + non-negative, else `ValueError`.
- `select_region(..., flow_magnitudes=None)`: when supplied, `importance = clamp(flow_magnitudes, min=1e-6)`
  replaces the prev-noise variance metric; the prev-noise fallback and first-step L2-norm path are
  unchanged, so existing callers (no flow tensor) behave identically.
- `forward(..., flow_magnitudes=None, starvation_scale=0.5)`: validates flow in the auto-select branch
  and threads both into `select_region` (`starvation_scale` → existing `k_starvation`).
- `update_skip_record` now uses **true starvation accounting**: `skip_k.add_(1)` then zero only the
  selected tokens — selected → 0, unselected → +1/step. (Old code zeroed-then-incremented, leaving
  selected tokens at 1.) This is a global change to the shared function and applies to the non-flow
  path too. Called ONCE per step from the positive CFG branch; the negative branch reuses the exact
  positive selection. Dense all-token updates zero `skip_k` in `forward` (every patch serviced).

### RAFT weight-download behavior
`raft_small(weights=Raft_Small_Weights.DEFAULT)` (== `C_T_V2`) downloads ~4 MB on first use from
download.pytorch.org. Run in FP32/`.eval()`/`torch.inference_mode()`. The torchvision model does NOT
self-normalize — apply `Raft_Small_Weights.DEFAULT.transforms()` (maps [0,1] → [-1,1]) before each
pair batch. Take `flows[-1]` (final of the 12 internal updates).

### GPU-memory / offload sequence (79 GB box)
1. Encode prompts (T5 on GPU) → move `text_encoder` + `vae` (+ image_encoder/motion_controller) to CPU.
2. Dense reference pass (DiT only on GPU) → move VAE back to GPU → decode → save reference → keep the
   RGB frames as float32 [0,1] numpy arrays.
3. Move VAE back to CPU; load RAFT-Small on GPU (tiny); estimate flow over all pairs in `raft_microbatch`
   microbatches, freeing per-microbatch temporaries.
4. `del` reference frames + RAFT model, `torch.cuda.empty_cache()` + `synchronize()`.
5. RAS pass (DiT + flow magnitudes on GPU) → move VAE back → decode `ras_flow_ranked.mp4`.

### Running
- `python examples/wanvideo/model_inference/RAS-Wan2.1-T2V-1.3B-OpticalFlow.py` → `wan_flow_reference.mp4`
  + `ras_flow_ranked.mp4`.
- `--smoke`: 17 frames, 64×128, 3 steps, 1 dense step, `vae_tiled=False` — prints RAFT flow shape,
  initial-latent equality assertion, per-step selected-token counts, and both output paths.
- CPU unit tests: `python tests/test_ras_flow_bootstrap.py`.
