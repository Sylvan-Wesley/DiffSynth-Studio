"""CPU unit tests for the MotionCache token-selection branch.

Covers the SkyReels-following ``select_region`` MotionCache heuristic in
``diffsynth/models/wan_video_dit.py``:

- phase-1: uniform weights (W = 1) for the first ``mc_phase1_steps`` sparse steps
  (SkyReels ``token_phase1_update_count``);
- motion map W (SkyReels ``_compute_weights_from_frame_diff``): per-spatial-token
  relative-L1 inter-frame difference of the previous prediction, frame 0 reuses
  frame 1, per-frame ``weight_norm_mode`` normalization ("mean" default);
- drift (SkyReels ``_compute_per_token_distance``, no polynomial): per-frame
  relative-L1 between consecutive steps' predictions, broadcast to the frame's
  spatial tokens;
- accumulation A += W * drift and TOP-K selection (RAS override of SkyReels'
  ``A >= threshold``), with the accumulator reset at selected tokens via
  ``reset_motion_accumulator``.

Plain-assert runner (no pytest dependency)::

    python tests/test_motion_cache.py
"""

import math
import os
import sys

import torch

# Make the repo root importable when running `python tests/test_motion_cache.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffsynth.models.wan_video_dit import WanModel  # noqa: E402


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


def mc_heuristics(**over):
    cfg = {
        "name": "MotionCache",
        "weight_norm_mode": "mean",
        "weight_floor": 0.3,
        "mc_phase1_steps": 3,
    }
    cfg.update(over)
    return cfg


# ═══════════════════════════════════════════════════════════════
# 1. Phase-1: uniform weights + per-frame drift broadcast
# ═══════════════════════════════════════════════════════════════

def test_phase1_uniform_weights_and_per_frame_drift():
    """Phase-1 uses W = 1, so A equals the per-frame drift broadcast to the
    frame's spatial tokens. Frame 0 sees no change -> drift 0; frame 1 moved by
    1.0 -> uniform positive drift."""
    model = tiny_model()
    B, F, h, w = 1, 3, 2, 2
    S = F * h * w
    hw = h * w
    C = 64
    P = torch.randn(B, S, C)
    Q = P.clone()
    Q[:, hw:2 * hw, :] += 1.0            # frame 1 prediction moves by 1.0
    model.A = torch.zeros(B, S)
    model._prev_noise_tokens = P         # seeded, like the RAS script after warm-up
    model._mc_phase1_count = 0
    sel = model.select_region(
        x=torch.randn(B, S, 64), ratio=0.5, timestep=torch.tensor([1.0]),
        context=None, skip_list=torch.zeros(B, S), skip_k=torch.zeros(B, S),
        prev_noise_tokens=Q, grid=(F, h, w),
        use_heuristics=mc_heuristics(),
    )
    A = model.A
    # Phase-1 consumed one step; W = 1 -> A == drift (broadcast per frame).
    assert model._mc_phase1_count == 1
    assert torch.allclose(A[0, :hw], torch.zeros(hw))          # frame 0: no change
    assert torch.allclose(A[0, 2 * hw:], torch.zeros(hw))      # frame 2: no change
    assert (A[0, hw:2 * hw] > 0).all()                         # frame 1: moved
    # Per-frame broadcast: every spatial token of a frame has the same A.
    assert torch.allclose(A[0, hw:2 * hw], A[0, hw:2 * hw].mean().expand(hw))
    # Top-k returns exactly ceil(S * ratio) tokens.
    assert sel.shape == (1, math.ceil(S * 0.5))
    assert sel[0].max().item() < S


def test_topk_selects_highest_accumulated_tokens():
    """Top-k by A: with only frame 1 accumulating drift, the top ceil(S/3) tokens
    are exactly the four frame-1 spatial tokens."""
    model = tiny_model()
    B, F, h, w = 1, 3, 2, 2
    S = F * h * w
    hw = h * w
    C = 64
    P = torch.randn(B, S, C)
    Q = P.clone()
    Q[:, hw:2 * hw, :] += 1.0
    model.A = torch.zeros(B, S)
    model._prev_noise_tokens = P
    model._mc_phase1_count = 0
    sel = model.select_region(
        x=torch.randn(B, S, 64), ratio=1.0 / 3.0, timestep=torch.tensor([1.0]),
        context=None, skip_list=torch.zeros(B, S), skip_k=torch.zeros(B, S),
        prev_noise_tokens=Q, grid=(F, h, w),
        use_heuristics=mc_heuristics(),
    )
    assert set(sel[0].tolist()) == set(range(hw, 2 * hw))


# ═══════════════════════════════════════════════════════════════
# 2. Motion map (phase-2): temporal frame difference, frame-0 reuse
# ═══════════════════════════════════════════════════════════════

def test_phase2_motion_weights_temporal_axis_and_frame0_reuse():
    """With mc_phase1_steps=0 (phase-2 immediately) and a uniform drift (prev =
    0.5*Q), A reveals W directly:
    - the motion map is a TEMPORAL frame difference: a hotspot at spatial 0 in
      frame 1 lands at the SAME spatial position in every frame (frame 0 via
      reuse), NOT on flattened spatial neighbours;
    - frame 0 reuses frame 1's motion map -> identical A across those frames."""
    model = tiny_model()
    B, F, h, w = 1, 3, 2, 2
    S = F * h * w
    hw = h * w
    C = 64
    big = 5.0
    # Ones everywhere except frame 1 spatial 0 = big (a temporal hotspot).
    Q = torch.ones(B, S, C)
    Q[:, hw + 0, :] = big
    model.A = torch.zeros(B, S)
    model._prev_noise_tokens = (0.5 * Q).detach()   # uniform per-frame drift ~= 1
    model._mc_phase1_count = 0
    sel = model.select_region(
        x=torch.randn(B, S, 64), ratio=1.0 / S, timestep=torch.tensor([1.0]),
        context=None, skip_list=torch.zeros(B, S), skip_k=torch.zeros(B, S),
        prev_noise_tokens=Q, grid=(F, h, w),
        use_heuristics=mc_heuristics(mc_phase1_steps=0),
    )
    A = model.A
    # Frame 0 reuses frame 1's motion map, and drift is uniform -> A matches.
    assert torch.allclose(A[0, :hw], A[0, hw:2 * hw], atol=1e-4)
    # The hotspot token has the highest A in its frame.
    assert A[0, hw + 0] > A[0, hw + 1]
    # Temporal axis: high A appears only at spatial position 0 across frames
    # (indices 0, 4, 8), never at spatial neighbours (1, 5, 9).
    spatial0 = {0, hw, 2 * hw}
    assert set(sel[0].tolist()) <= spatial0
    assert all(A[0, s] > A[0, s + 1] for s in (0, hw, 2 * hw))


def test_phase2_motion_weights_recomputed_each_step():
    """W is recomputed from the CURRENT prev_noise_tokens every step (SkyReels
    cadence): changing the frame structure between steps changes the resulting A,
    so weights are not cached across calls."""
    model = tiny_model()
    B, F, h, w = 1, 3, 2, 2
    S = F * h * w
    hw = h * w
    C = 64
    heur = mc_heuristics(mc_phase1_steps=0)
    model.A = torch.zeros(B, S)
    model._mc_phase1_count = 0
    # Step 1: no cached prev -> drift 0, A stays 0.
    Q1 = torch.ones(B, S, C)
    model._prev_noise_tokens = None
    model.select_region(
        x=torch.randn(B, S, 64), ratio=0.5, timestep=torch.tensor([1.0]),
        context=None, skip_list=torch.zeros(B, S), skip_k=torch.zeros(B, S),
        prev_noise_tokens=Q1, grid=(F, h, w), use_heuristics=heur,
    )
    assert model.A.abs().sum().item() == 0.0      # no prev -> no drift
    # Step 2: a frame-1 temporal hotspot; prev = 0.5*Q2 isolates W (uniform drift).
    Q2 = torch.ones(B, S, C)
    Q2[:, hw + 0, :] = 3.0                        # frame 1 spatial 0 hotspot
    model._prev_noise_tokens = (0.5 * Q2).detach()
    model.select_region(
        x=torch.randn(B, S, 64), ratio=0.5, timestep=torch.tensor([1.0]),
        context=None, skip_list=torch.zeros(B, S), skip_k=torch.zeros(B, S),
        prev_noise_tokens=Q2, grid=(F, h, w), use_heuristics=heur,
    )
    A = model.A
    # The hotspot accumulates the most A in its frame, and it is > 0.
    assert A[0, hw + 0] > A[0, hw + 1]
    assert A[0, hw + 0] > 0.0


# ═══════════════════════════════════════════════════════════════
# 3. First step (no cached prev) and reset semantics
# ═══════════════════════════════════════════════════════════════

def test_first_step_unseeded_zero_drift():
    """With _prev_noise_tokens = None (no-CFG paths, or a fresh model) the drift
    is zero, A stays zero, and top-k deterministically returns the first k tokens.
    Top-k never returns an empty selection."""
    model = tiny_model()
    B, F, h, w = 1, 3, 2, 2
    S = F * h * w
    C = 64
    model.A = torch.zeros(B, S)
    model._prev_noise_tokens = None
    model._mc_phase1_count = 0
    sel = model.select_region(
        x=torch.randn(B, S, 64), ratio=0.5, timestep=torch.tensor([1.0]),
        context=None, skip_list=torch.zeros(B, S), skip_k=torch.zeros(B, S),
        prev_noise_tokens=torch.randn(B, S, C), grid=(F, h, w),
        use_heuristics=mc_heuristics(),
    )
    assert model.A.abs().sum().item() == 0.0
    assert sel.shape == (1, math.ceil(S * 0.5))
    assert set(sel[0].tolist()) == set(range(math.ceil(S * 0.5)))


def test_reset_accumulator_zeroes_selected_only():
    """reset_motion_accumulator zeroes the just-computed tokens and leaves the
    rest intact; None accumulator is a no-op."""
    model = tiny_model()
    B, S = 1, 12
    model.A = torch.arange(S, dtype=torch.float32).unsqueeze(0)
    sel = torch.tensor([[2, 5, 9]])
    model.reset_motion_accumulator(sel)
    assert torch.equal(model.A[0, sel[0]], torch.zeros(3))
    keep = torch.ones(B, S, dtype=torch.bool)
    keep[0, sel[0]] = False
    assert torch.equal(model.A[keep], torch.arange(S)[keep[0]].float())
    model.A = None
    model.reset_motion_accumulator(sel)   # must not raise


# ═══════════════════════════════════════════════════════════════
# 4. Validation errors
# ═══════════════════════════════════════════════════════════════

def _expect(fn, exc_type):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def test_motion_cache_validation():
    model = tiny_model()
    B, F, h, w = 1, 3, 2, 2
    S = F * h * w
    C = 64
    x = torch.randn(B, S, 64)
    prev = torch.randn(B, S, C)
    skip = torch.zeros(B, S)
    t = torch.tensor([1.0])

    def call(**over):
        kw = dict(prev_noise_tokens=prev, grid=(F, h, w),
                  use_heuristics=mc_heuristics())
        kw.update(over)
        model._prev_noise_tokens = None
        model._mc_phase1_count = 0
        return model.select_region(x, 0.5, t, None, skip, skip, **kw)

    model.A = torch.zeros(B, S)
    _expect(lambda: call(use_heuristics="MotionCache"), ValueError)          # dict required
    _expect(lambda: call(prev_noise_tokens=None), ValueError)                # needs prev
    _expect(lambda: call(grid=None), ValueError)                             # needs grid
    _expect(lambda: call(use_heuristics=mc_heuristics(weight_norm_mode="bad")), AssertionError)
    _expect(lambda: call(use_heuristics=mc_heuristics(weight_floor=-0.1)), AssertionError)
    model.A = None
    _expect(lambda: call(), AssertionError)                                  # A not initialized


# ═══════════════════════════════════════════════════════════════
# 5. End-to-end through WanModel.forward (gather/scatter + KV cache)
# ═══════════════════════════════════════════════════════════════

def test_forward_motion_cache_sparse_step():
    model = tiny_model()
    B, C_in = 1, 16
    F, H, W = 3, 8, 8                       # patched (3, 4, 4) -> S = 48
    S = F * (H // 2) * (W // 2)
    x = torch.randn(B, C_in, F, H, W)
    t = torch.tensor([1.0])
    ctx = torch.randn(B, 6, 8)              # text_dim = 8
    kv = [{} for _ in range(2)]
    ctx_kv = [{} for _ in range(2)]
    skip_list = torch.zeros(B, S)
    skip_k = torch.zeros(B, S)
    all_patches = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)

    # Dense warm-up seeds the KV caches and produces the first noise tokens.
    _, nt = model.forward(
        x, t, ctx,
        skip_list=skip_list, skip_k=skip_k,
        kv_cache=kv, ctx_kv_cache=ctx_kv,
        selected_patches=all_patches, ratio=0.25,
        return_noise_tokens=True,
    )
    assert nt.shape == (B, S, 64)

    # Sparse MotionCache step: A initialized, heuristic dict, reset via the
    # public method exactly like the RAS script does.
    model.A = torch.zeros(B, S)
    model._prev_noise_tokens = None
    model._mc_phase1_count = 0
    ratio = 0.25
    out, nt2 = model.forward(
        x, t, ctx,
        skip_list=skip_list, skip_k=skip_k,
        kv_cache=kv, ctx_kv_cache=ctx_kv,
        ratio=ratio, dumb_update="Previous",
        prev_noise_tokens=nt, dumb_noise_tokens=nt,
        use_heuristics=mc_heuristics(),
        return_noise_tokens=True,
    )
    assert out.shape == x.shape
    assert nt2.shape == (B, S, 64)
    sel = model.get_last_selected_patches()
    # Top-k returns exactly ceil(S * ratio) tokens (never empty).
    assert sel.shape == (1, math.ceil(S * ratio))
    assert 0 <= sel[0].min().item() < sel[0].max().item() < S

    model.reset_motion_accumulator(sel)
    assert torch.equal(model.A[0, sel[0]], torch.zeros_like(model.A[0, sel[0]]))
    assert (model.A >= 0).all()


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
