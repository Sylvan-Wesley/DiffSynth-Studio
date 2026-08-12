# AGENT.md — RAS Implementation (DiffSynth-Studio)

Training-free inference acceleration for **Wan2.1-T2V-1.3B** via Region-Adaptive Sampling
(Liu et al., "Region-Adaptive Sampling for Diffusion Transformers", CVPR 2026). Each denoising
step runs only a fraction (`ratio`) of video-latent tokens through the full DiT stack; the rest
get a cheap "dumb" update. Goal: ~2x speedup with minimal quality loss.

All RAS work lives on the local `ras` branch. **`origin/main` is the upstream modelscope repo
and contains NO RAS code** — never base RAS changes on it.

## Files
- `examples/wanvideo/model_inference/RAS-Wan2.1-T2V-1.3B.py` — main inference script. Owns all
  RAS state (KV caches, skip counters, dense warm-up), runs CFG, decodes. Config knobs at top.
- `examples/wanvideo/model_inference/benchmark_RAS_vs_Wan.py` — times full inference
  (`kv_cache=None`) vs RAS (`kv_cache`+`ratio`) through `WanModel.forward()` directly.
- `examples/wanvideo/model_inference/visualize_ras_masks.py` — (untracked; working copy only)
  heatmap overlay of per-step selection masks.
- `diffsynth/models/wan_video_dit.py` — ALL RAS machinery (module helpers + `WanModel` methods).

## Structure of `diffsynth/models/wan_video_dit.py` (grep-verified line refs)
Module helpers:
- `flash_attention` (30) — FA3/FA2/Sage dispatch.
- `gather_tokens` (66) / `scatter_tokens_` (75) — gather/scatter token rows by `[B,N]` indices.
- `cache_ready` (87) — is the KV cache populated?
- `selection_mask_to_grid` (94) — `[B,S]` bool mask → `[B,1,f,h,w]` grid for viz.
- `rope_apply` (139) — RoPE, real-valued bf16 (see Gotchas).
- `set_to_torch_norm` (184) — switches RMSNorm to fused `F.rms_norm`.

Model:
- `SelfAttention.forward` (239) — with cache + `selected_patches`: compute q for active tokens;
  compute k/v for active and **scatter back** into full cached tensors; inactive read stale k/v.
- `CrossAttention.forward` (291) — ctx K/V cached per layer; q computed only for active tokens.
- `DiTBlock.forward` (359) — threads `selected_patches` + both caches through self/cross/ffn.
- `Head.forward` (407) — outputs token-level noise `[B, S, out_dim*prod(patch_size)]`.
- `WanModel`: `_prev_noise_tokens` (552); `patchify` (629); `unpatchify` (645);
  `select_region` (652); `update_skip_record` (700); `get_selection_masks` (716);
  `clear_selection_masks` (723); `forward` (727; RAS region-selection block begins ~line 762).

## Per-step data flow
1. **Dense warm-up** (`num_dense_steps`, default 3) runs ALL tokens — seeds per-layer KV caches
   and `_prev_noise_tokens`. Sparse steps follow.
2. **Sparse step** — `select_region` picks top-k tokens:
   `importance = 1/(std(prev_noise)+eps)` (low variance = semantically settled = safe to skip),
   `score = importance * exp(k_starvation * skip_k)` (starvation prevention). First step has no
   prev noise → falls back to L2 norm of latents.
3. `update_skip_record` — reset selected tokens' drop counter to 0, then increment all
   (unselected accumulate, selected → 1).
4. DiT blocks run ONLY on gathered active tokens (see attention notes above).
5. **Head / dumb update**: active → fresh prediction; inactive → **reuse `_prev_noise_tokens`**
   (carry forward the last real prediction). `head()` is only valid on DiT-block OUTPUT, so the
   old `head(raw_input)` path is a fallback only, used when no prior prediction exists. Store the
   mixed full-sequence prediction back to `_prev_noise_tokens` (drives next step's selection AND
   dumb reuse).
6. Unpatchify → `noise_pred` → `scheduler.step`.
7. **CFG** runs TWO forwards (posi/nega) sharing `skip_list`/`skip_k` and one `_prev_noise_tokens`
   buffer — both branches select the SAME regions. Known nuance: posi's inactive tokens reuse
   nega's stale prediction and vice versa (valid CFG mix of stale preds). Per-branch buffers =
   possible follow-up.

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
- `_prev_noise_tokens` is reset to `None` before the loop; sparse steps with no prior prediction
  hit the head-on-raw fallback (keep it working).
- KV caches persist across steps → dense warm-up required before sparse steps.
- Eval-only; `inference_mode` forbids autograd.

## Running
- Model: `Wan-AI/Wan2.1-T2V-1.3B`. Defaults: `ratio=0.25`, `num_dense_steps=3`,
  `num_inference_steps=50`, `cfg_scale=5.0`, 81 frames @ 480×832.
- `python examples/wanvideo/model_inference/RAS-Wan2.1-T2V-1.3B.py` → `ras_output.mp4`; with
  `enable_viz=True`, per-step mask `.npy` files under `ras_masks/`.
