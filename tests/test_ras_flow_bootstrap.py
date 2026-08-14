"""CPU unit tests for the flow-guided RAS experiment.

Covers the pure pooling/grouping helpers in
``examples/wanvideo/model_inference/RAS-Wan2.1-T2V-1.3B-OpticalFlow.py`` and the optional
flow-ranking inputs added to ``diffsynth/models/wan_video_dit.py``:

- flow magnitude -> [B, S] pooling and the causal 1-then-4 temporal grouping;
- ranking order for known flow magnitudes and starvation counts;
- zero-motion handling via the score floor (starvation can still recover static regions);
- true starvation accounting + positive/negative CFG selection sharing (single counter update);
- unchanged fallback behavior when flow_magnitudes is None.

Plain-assert runner (no pytest dependency)::

    python tests/test_ras_flow_bootstrap.py
"""

import importlib.util
import math
import os
import sys

import torch

# Make the repo root importable when running `python tests/test_ras_flow_bootstrap.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffsynth.models.wan_video_dit import (  # noqa: E402
    WanModel,
    validate_flow_magnitudes,
)

SCRIPT_PATH = os.path.join(
    REPO_ROOT,
    "examples", "wanvideo", "model_inference",
    "RAS-Wan2.1-T2V-1.3B-OpticalFlow.py",
)

_script = None


def script():
    """Lazily import the experiment script (its filename has dashes -> importlib)."""
    global _script
    if _script is None:
        spec = importlib.util.spec_from_file_location("ras_flow_script", SCRIPT_PATH)
        _script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_script)
    return _script


def tiny_model() -> WanModel:
    """A tiny WanModel that exercises the same RAS code paths on CPU."""
    model = WanModel(
        dim=32,
        in_dim=16,          # matches Wan's VAE z_dim (in_dim == out_dim)
        ffn_dim=64,
        out_dim=16,
        text_dim=8,
        freq_dim=64,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=4,
        num_layers=2,
        has_image_input=False,
    )
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════
# 1. Pooling + causal 1-then-4 temporal grouping + [B, S] flatten
# ═══════════════════════════════════════════════════════════════

def test_temporal_grouping_causal():
    m = script()
    # 16 flow pairs, magnitudes 1..16; 5 latent frames (T_RGB = 4*5 - 3 = 17 frames -> 16 pairs).
    g = torch.arange(1, 17, dtype=torch.float32).view(16, 1, 1)
    out = m.temporal_group_flow(g, 5)
    assert out.shape == (5, 1, 1)
    # latent frame 0 <- flow pair 0
    assert torch.allclose(out[0], torch.tensor([1.0]))
    # latent frame k>=1 <- pairs [4k-3, 4k]
    assert torch.allclose(out[1], torch.tensor([3.5]))    # mean(2,3,4,5)
    assert torch.allclose(out[2], torch.tensor([7.5]))    # mean(6,7,8,9)
    assert torch.allclose(out[3], torch.tensor([11.5]))   # mean(10,11,12,13)
    # final partial group averaged normally
    assert torch.allclose(out[4], torch.tensor([15.0]))   # mean(14,15,16)


def test_spatial_pool_and_b_s_flatten():
    m = script()
    mag = torch.rand(3, 64, 128)
    pooled = m.spatial_pool_to_patch_grid(mag, (4, 8))
    assert pooled.shape == (3, 4, 8)
    # exact 16x16 block mean (adaptive pooling is exact when divisible)
    expected = mag.view(3, 4, 16, 8, 16).mean(dim=(2, 4))
    assert torch.allclose(pooled, expected, atol=1e-5)

    flat = m.flow_grid_to_b_s(pooled)
    assert flat.shape == (1, 3 * 4 * 8)
    # frame-major order (f then h then w), matching WanModel.patchify's `b (f h w) d`.
    assert torch.equal(flat[0, :4 * 8], pooled[0].reshape(-1))
    assert torch.equal(flat[0, 4 * 8:2 * 4 * 8], pooled[1].reshape(-1))
    assert torch.equal(flat[0, 2 * 4 * 8:], pooled[2].reshape(-1))


# ═══════════════════════════════════════════════════════════════
# 2 + 3. Ranking order + zero-motion floor
# ═══════════════════════════════════════════════════════════════

def test_ranking_order_and_zero_motion_floor():
    model = tiny_model()
    B, S = 1, 20
    x = torch.randn(B, S, 32)
    skip_list = torch.zeros(B, S)

    flow = torch.zeros(B, S)
    flow[0, :5] = 10.0          # tokens 0..4 high motion
    flow[0, 5:] = 0.0           # tokens 5..19 static (zero flow)

    # No starvation: high-motion tokens are always selected first.
    skip_k = torch.zeros(B, S)
    sel = model.select_region(
        x, 0.5, torch.tensor([1.0]), None, skip_list, skip_k,
        flow_magnitudes=flow, k_starvation=1.0,
    )
    assert sel.shape == (B, math.ceil(S * 0.5))
    sel_set = set(sel[0].tolist())
    assert set(range(5)).issubset(sel_set)          # all high-motion selected
    assert len(sel_set) == 10

    # Zero-motion floor: after enough starvation, static regions are selected too.
    # score_static = max(0, 1e-6) * exp(1.0 * 50) >> score_high = 10 * exp(0).
    skip_k = torch.zeros(B, S)
    skip_k[0, 5:] = 50.0
    sel2 = model.select_region(
        x, 0.5, torch.tensor([1.0]), None, skip_list, skip_k,
        flow_magnitudes=flow, k_starvation=1.0,
    )
    sel2_set = set(sel2[0].tolist())
    assert len(sel2_set) == 10
    assert all(i >= 5 for i in sel2_set)            # only static regions recovered
    # The multiplicative term stayed effective despite zero flow (score floor).
    assert any(i >= 5 for i in sel2_set)


# ═══════════════════════════════════════════════════════════════
# 4. True starvation accounting + posi/nega sharing
# ═══════════════════════════════════════════════════════════════

def test_starvation_accounting_true():
    model = tiny_model()
    B, S = 1, 10
    skip_list = torch.zeros(B, S)
    skip_k = torch.zeros(B, S)
    sel = torch.tensor([[0, 1, 2]])
    model.update_skip_record(skip_list, skip_k, sel)
    # selected tokens reset to zero; ONLY unselected increment.
    assert skip_k[0, :3].tolist() == [0, 0, 0]
    assert skip_k[0, 3:].tolist() == [1] * 7

    # A dense all-token update resets every starvation count.
    skip_k = torch.zeros(B, S)
    model.update_skip_record(skip_list, skip_k, torch.arange(S).unsqueeze(0))
    assert skip_k.tolist() == [[0] * S]


def test_forward_flow_guided_and_selection_sharing():
    model = tiny_model()
    B, C = 1, 16                           # Wan VAE z_dim
    F, H, W = 2, 8, 8                       # patched to (2, 4, 4) -> S = 32
    S = F * (H // 2) * (W // 2)
    x = torch.randn(B, C, F, H, W)
    t = torch.tensor([1.0])
    ctx = torch.randn(B, 6, 8)              # text_dim = 8
    ratio = 0.25                           # ceil(32 * 0.25) = 8 tokens

    flow = torch.zeros(B, S)
    flow[0, :8] = 5.0                      # first 8 tokens high motion

    posi_kv = [{} for _ in range(2)]
    posi_ctx_kv = [{} for _ in range(2)]
    nega_kv = [{} for _ in range(2)]
    nega_ctx_kv = [{} for _ in range(2)]
    skip_list = torch.zeros(B, S)
    skip_k = torch.zeros(B, S)
    all_patches = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)

    # Dense warm-up for BOTH branches seeds the per-condition KV caches (mirrors
    # the real script lifecycle; sparse steps require ready caches).
    _, nt = model.forward(
        x, t, ctx,
        skip_list=skip_list, skip_k=skip_k,
        kv_cache=posi_kv, ctx_kv_cache=posi_ctx_kv,
        selected_patches=all_patches, ratio=ratio,
        return_noise_tokens=True,
    )
    _, _ = model.forward(
        x, t, ctx,
        skip_list=skip_list, skip_k=skip_k,
        kv_cache=nega_kv, ctx_kv_cache=nega_ctx_kv,
        selected_patches=all_patches, ratio=ratio,
        return_noise_tokens=True,
    )
    assert nt.shape == (B, S, 64)           # out_dim(16) * prod(patch_size)(4)
    assert skip_k.tolist() == [[0] * S]     # dense passes serviced every patch

    # Sparse flow-guided step (positive branch auto-selects).
    out2, nt2 = model.forward(
        x, t, ctx,
        skip_list=skip_list, skip_k=skip_k,
        kv_cache=posi_kv, ctx_kv_cache=posi_ctx_kv,
        ratio=ratio, dumb_update="Previous",
        prev_noise_tokens=nt, dumb_noise_tokens=nt,
        flow_magnitudes=flow, starvation_scale=1.0,
        return_noise_tokens=True,
    )
    assert out2.shape == x.shape
    sel = model.get_last_selected_patches()
    assert sel.shape == (B, math.ceil(S * ratio))
    assert set(sel[0].tolist()) == set(range(8))    # exactly the high-motion tokens
    # True accounting after the sparse step: selected -> 0, unselected -> +1.
    assert skip_k[0, :8].tolist() == [0] * 8
    assert skip_k[0, 8:].tolist() == [1] * 24

    # Negative branch REUSES the exact positive selection: no re-selection, and
    # crucially the starvation counter is NOT updated a second time.
    skip_k_before = skip_k.clone()
    out3, nt3 = model.forward(
        x, t, ctx,
        skip_list=skip_list, skip_k=skip_k,
        kv_cache=nega_kv, ctx_kv_cache=nega_ctx_kv,
        selected_patches=sel, ratio=ratio, dumb_update="Previous",
        prev_noise_tokens=nt2, dumb_noise_tokens=nt2,
        return_noise_tokens=True,
    )
    assert out3.shape == x.shape
    assert torch.equal(skip_k, skip_k_before)       # single starvation-counter update


# ═══════════════════════════════════════════════════════════════
# 5. Fallback when flow_magnitudes is None
# ═══════════════════════════════════════════════════════════════

def test_fallback_without_flow():
    model = tiny_model()
    B, S = 1, 20
    x = torch.randn(B, S, 32)
    skip_list = torch.zeros(B, S)
    skip_k = torch.zeros(B, S)

    # Previous-noise metric: low-std tokens (more confident) rank first.
    prev = torch.randn(B, S, 32)
    prev[0, :5] *= 0.01
    sel = model.select_region(
        x, 0.25, torch.tensor([1.0]), None, skip_list, skip_k,
        prev_noise_tokens=prev, k_starvation=0.0,
    )
    assert sel.shape == (B, math.ceil(S * 0.25))
    assert set(sel[0].tolist()) == {0, 1, 2, 3, 4}

    # End-to-end: sparse forward with flow_magnitudes=None still selects and returns.
    F, H, W = 2, 8, 8
    S_fwd = F * (H // 2) * (W // 2)         # patched token count (32), distinct from S above
    x5 = torch.randn(B, 16, F, H, W)
    ctx = torch.randn(B, 6, 8)
    kv = [{} for _ in range(2)]
    ctx_kv = [{} for _ in range(2)]
    skip_list = torch.zeros(B, S_fwd)
    skip_k = torch.zeros(B, S_fwd)
    # Dense warm-up seeds the KV caches before the sparse step (real-life ordering).
    all_patches = torch.arange(S_fwd, dtype=torch.long).unsqueeze(0).expand(B, -1)
    model.forward(
        x5, torch.tensor([1.0]), ctx,
        skip_list=skip_list, skip_k=skip_k,
        kv_cache=kv, ctx_kv_cache=ctx_kv,
        selected_patches=all_patches, ratio=0.25,
        return_noise_tokens=True,
    )
    skip_k.zero_()
    out, nt = model.forward(
        x5, torch.tensor([1.0]), ctx,
        skip_list=skip_list, skip_k=skip_k,
        kv_cache=kv, ctx_kv_cache=ctx_kv,
        ratio=0.25, dumb_update="Previous",
        prev_noise_tokens=None, dumb_noise_tokens=None,
        return_noise_tokens=True,
    )
    assert out.shape == x5.shape
    assert nt.shape == (B, S_fwd, 64)
    assert model.get_last_selected_patches().shape == (B, math.ceil(0.25 * S_fwd))


# ═══════════════════════════════════════════════════════════════
# validate_flow_magnitudes
# ═══════════════════════════════════════════════════════════════

def test_validate_flow_magnitudes():
    x = torch.randn(2, 10, 8)
    ok = validate_flow_magnitudes(torch.rand(2, 10), x)
    assert ok.shape == (2, 10)
    for bad in [
        torch.rand(2, 9),                                # wrong S
        torch.rand(3, 10),                               # wrong B
        torch.full((2, 10), float("nan")),               # non-finite
        -torch.rand(2, 10),                              # negative
    ]:
        try:
            validate_flow_magnitudes(bad, x)
        except ValueError:
            pass
        else:
            raise AssertionError(f"validate_flow_magnitudes should reject {tuple(bad.shape)}")


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

def main():
    tests = [
        fn for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
