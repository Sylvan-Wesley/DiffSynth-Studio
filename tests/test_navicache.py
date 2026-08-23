"""CPU unit tests for the NaviCache whole-forward residual cache.

Covers ``diffsynth/models/navicache.py`` (``NaviCache``):

- alignment steps always compute and populate the residual + Kalman state;
- a post-alignment skip reuses the cached residual exactly (``input + residual``)
  without running the transformer;
- ``thresh=0`` forces a compute every step;
- CFG pairing: the skip/compute decision is made on the conditional branch only,
  and the unconditional branch reuses its OWN residual;
- the final pair is always computed exactly.

Plain-assert runner (no pytest dependency)::

    python tests/test_navicache.py
"""

import os
import sys

import torch

# Make the repo root importable when running `python tests/test_navicache.py`.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from diffsynth.models.navicache import NaviCache  # noqa: E402
from diffsynth.models.wan_video_dit import WanModel  # noqa: E402


def tiny_model() -> WanModel:
    """A tiny WanModel with in_dim == out_dim (like Wan's VAE z_dim = 16)."""
    model = WanModel(
        dim=32,
        in_dim=16,
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


def make_inputs(model, seed):
    """A latent/timestep/context triple with the shapes ``WanModel.forward`` needs."""
    torch.manual_seed(seed)
    B, C, F, H, W = 1, 16, 3, 8, 8
    x = torch.randn(B, C, F, H, W)
    t = torch.tensor([1.0])
    ctx = torch.randn(B, 6, 8)
    return x, t, ctx


def run_alignment(navi, steps):
    """Run ``steps`` CFG pairs (cond + uncond) through the wrapper."""
    outputs = []
    for s in range(steps):
        x, t, ctx = make_inputs(navi.model, 1000 + s)
        cond = navi.forward(x, t, ctx)
        uncond = navi.forward(x, t, ctx)
        outputs.append((cond, uncond))
    return outputs


# ═══════════════════════════════════════════════════════════════
# 1. Alignment populates residual + Kalman state
# ═══════════════════════════════════════════════════════════════

def test_alignment_populates_state():
    model = tiny_model()
    navi = NaviCache(model, thresh=0.05, align_steps=2, num_inference_steps=5, cfg=True)
    run_alignment(navi, 2)  # 2 steps = 4 forwards = align_forwards
    assert navi.forward_count == 4
    assert navi.cond_residual is not None
    assert navi.uncond_residual is not None
    assert navi.state_ratio is not None
    assert navi.prediction_ratio is not None
    assert navi.cond_residual.shape == (1, 16, 3, 8, 8)


# ═══════════════════════════════════════════════════════════════
# 2. Skip reuses the cached residual exactly (no transformer run)
# ═══════════════════════════════════════════════════════════════

def test_skip_reuses_residual():
    model = tiny_model()
    navi = NaviCache(model, thresh=float("inf"), align_steps=2, num_inference_steps=5, cfg=True)
    run_alignment(navi, 2)
    residual_before = navi.cond_residual.clone()

    # Post-alignment cond forward (forward_count 4 < cutoff 8): thresh=inf means
    # accumulated_error can never reach the threshold -> must skip.
    x, t, ctx = make_inputs(model, 9999)
    out = navi.forward(x, t, ctx)

    assert torch.allclose(out, x + residual_before)
    assert navi.forward_count == 5
    # The residual is untouched by the skip.
    assert torch.allclose(navi.cond_residual, residual_before)


# ═══════════════════════════════════════════════════════════════
# 3. thresh=0 forces a compute every step
# ═══════════════════════════════════════════════════════════════

def test_zero_thresh_forces_compute():
    model = tiny_model()
    navi = NaviCache(model, thresh=0.0, align_steps=2, num_inference_steps=5, cfg=True)
    run_alignment(navi, 2)

    x, t, ctx = make_inputs(model, 5555)
    out = navi.forward(x, t, ctx)

    # Compute path: output must match a direct model call, and the residual must
    # be refreshed to (output - input).
    expected = navi.model(x, t, ctx)
    assert torch.allclose(out, expected)
    assert torch.allclose(navi.cond_residual, out - x)


# ═══════════════════════════════════════════════════════════════
# 4. CFG pairing: unconditional branch reuses its own residual
# ═══════════════════════════════════════════════════════════════

def test_cfg_pairing_uncond_uses_own_residual():
    model = tiny_model()
    navi = NaviCache(model, thresh=float("inf"), align_steps=2, num_inference_steps=5, cfg=True)
    # Align with DISTINCT contexts per CFG branch (as a real CFG loop does), so
    # the conditional and unconditional residuals differ.
    for s in range(2):
        x, t, _ = make_inputs(model, 1000 + s)
        navi.forward(x, t, torch.randn(1, 6, 8))   # cond context
        navi.forward(x, t, torch.randn(1, 6, 8))   # uncond context
    cond_res = navi.cond_residual.clone()
    uncond_res = navi.uncond_residual.clone()
    assert not torch.allclose(cond_res, uncond_res)  # sanity: residuals differ

    # Cond forward (skip) -> cond_residual reused.
    x4, t, _ = make_inputs(model, 7777)
    out_cond = navi.forward(x4, t, torch.randn(1, 6, 8))
    assert torch.allclose(out_cond, x4 + cond_res)

    # Uncond forward (paired skip) -> uncond_residual reused, NOT cond_residual.
    x5, t, _ = make_inputs(model, 8888)
    out_uncond = navi.forward(x5, t, torch.randn(1, 6, 8))
    assert torch.allclose(out_uncond, x5 + uncond_res)
    assert not torch.allclose(out_uncond, x5 + cond_res)


# ═══════════════════════════════════════════════════════════════
# 5. Final pair is always computed exactly
# ═══════════════════════════════════════════════════════════════

def test_final_pair_forced_compute():
    model = tiny_model()
    # num_inference_steps=5 -> num_forwards=10, cutoff_forwards=8.
    navi = NaviCache(model, thresh=float("inf"), align_steps=2, num_inference_steps=5, cfg=True)
    run_alignment(navi, 2)  # forwards 0..3 compute, forward_count=4

    # Forwards 4..7 skip (thresh=inf), advancing the counter to the cutoff.
    for s in range(4):
        x, t, ctx = make_inputs(model, 4000 + s)
        navi.forward(x, t, ctx)
    assert navi.forward_count == 8

    # Forward 8 (>= cutoff) must compute even though thresh=inf.
    x8, t, ctx = make_inputs(model, 9090)
    expected = navi.model(x8, t, ctx)
    out = navi.forward(x8, t, ctx)
    assert torch.allclose(out, expected)
    assert torch.allclose(navi.cond_residual, out - x8)


# ═══════════════════════════════════════════════════════════════
# 6. Single-forward (no CFG) mode
# ═══════════════════════════════════════════════════════════════

def test_single_forward_no_cfg_mode():
    model = tiny_model()
    # cfg=False: num_forwards=4, align_forwards=2, cutoff_forwards=3.
    navi = NaviCache(model, thresh=0.05, align_steps=2, num_inference_steps=4, cfg=False)
    for s in range(4):
        x, t, ctx = make_inputs(model, 3000 + s)
        out = navi.forward(x, t, ctx)
    assert navi.cond_residual is not None
    assert navi.state_ratio is not None
    # Every forward was treated as conditional: no uncond residual was created.
    assert navi.uncond_residual is None
    assert navi.forward_count == 0  # wrapped around after 4 forwards


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
