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
