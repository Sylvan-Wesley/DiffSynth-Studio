"""CPU tests for the per-patch CacheHead-vs-teacher error heat map."""

import torch

from cache_head_model import CacheHead, CacheHeadConfig, CacheHeadSchedule
from cache_head_error_heatmap import (
    collect_step_errors,
    render_error_heatmap,
    save_error_arrays,
    step_summary,
)
from cache_head_model_inference import full_step

from test_training_losses import GRID, LATENT_SHAPE, PATCH_SIZE, FakeDit, FakeScheduler


SCHEDULE = CacheHeadSchedule(15, (1, 2, 6, 10, 14))


def _collect(head=None, n_batch=1, seed=0):
    torch.manual_seed(seed)
    dit = FakeDit()
    scheduler = FakeScheduler(15)
    head = head if head is not None else CacheHead(CacheHeadConfig())
    return dit, scheduler, head, collect_step_errors(
        dit=dit, scheduler=scheduler, head=head, schedule=SCHEDULE, cfg_scale=5.0,
        patch_size=PATCH_SIZE, grid=GRID,
        latents=torch.randn(n_batch, *LATENT_SHAPE[1:]),
        ctx=torch.randn(n_batch, 4, 8),
        neg_ctx=torch.randn(1, 4, 8),
    )


def test_errors_have_token_grid_shape():
    _, _, _, result = _collect(n_batch=2)
    f, h, w = GRID
    assert result["errors"].shape == (15, 2, f, h, w)
    assert result["rel_errors"].shape == (15, 2, f, h, w)


def test_is_head_matches_schedule():
    _, _, _, result = _collect()
    for k in range(15):
        assert bool(result["is_head"][k]) == SCHEDULE.is_head_step(k)
    assert int(result["is_head"].sum()) == SCHEDULE.num_head_steps == 10


def test_no_head_prediction_before_the_first_full_step():
    _, _, _, result = _collect()
    # Step 0 is an anchor, so no previous guided tokens exist to build on.
    assert not bool(result["has_pred"][0])
    assert torch.isnan(result["errors"][0]).all()
    assert bool(result["has_pred"][1:].all())
    assert not torch.isnan(result["errors"][1:]).any()


def test_zero_init_head_reproduces_carry_previous_error():
    """A fresh head emits zero residual, so its prediction is exactly the
    carried tokens; the error must equal ||v_prev - v_teacher||."""
    torch.manual_seed(0)
    dit = FakeDit()
    scheduler = FakeScheduler(15)
    head = CacheHead(CacheHeadConfig(), zero_init_out_proj=True)
    latents = torch.randn(1, *LATENT_SHAPE[1:])
    ctx = torch.randn(1, 4, 8)
    neg_ctx = torch.randn(1, 4, 8)

    result = collect_step_errors(
        dit=dit, scheduler=scheduler, head=head, schedule=SCHEDULE, cfg_scale=5.0,
        patch_size=PATCH_SIZE, grid=GRID, latents=latents, ctx=ctx, neg_ctx=neg_ctx,
    )

    # Reproduce step 1's error by hand: carry from the step-0 anchor.
    t0 = scheduler.timesteps[0].reshape(1)
    with torch.no_grad():
        noise0, tokens0 = full_step(dit, latents, t0, ctx, neg_ctx, 5.0)
        latents1 = scheduler.step(noise0, t0, latents)
        t1 = scheduler.timesteps[1].reshape(1)
        _, teacher1 = full_step(dit, latents1, t1, ctx, neg_ctx, 5.0)
    expected = (tokens0 - teacher1).norm(dim=-1).reshape(1, *GRID)
    assert torch.allclose(result["errors"][1], expected, atol=1e-5)


def test_relative_error_is_scale_free():
    _, _, _, result = _collect()
    finite = result["rel_errors"][1:]
    assert torch.isfinite(finite).all()
    assert (finite >= 0).all()


def test_step_summary_is_nan_safe_and_per_step():
    _, _, _, result = _collect()
    summary = step_summary(result)
    assert len(summary["mean"]) == 15 and len(summary["p90"]) == 15
    # Step 0 has no prediction at all.
    assert summary["mean"][0] != summary["mean"][0]        # NaN
    assert all(v == v for v in summary["mean"][1:])


def test_render_writes_a_figure_and_arrays(tmp_path):
    _, _, _, result = _collect(n_batch=2)
    png = render_error_heatmap(result, tmp_path / "heat.png", title="test")
    pt = save_error_arrays(result, tmp_path / "heat.pt")
    assert png.is_file() and png.stat().st_size > 0
    assert pt.is_file()
    reloaded = torch.load(pt, weights_only=False)
    assert reloaded["errors"].shape == result["errors"].shape


def test_render_relative_and_single_frame(tmp_path):
    _, _, _, result = _collect()
    png = render_error_heatmap(result, tmp_path / "rel.png", relative=True, frame=0)
    assert png.is_file() and png.stat().st_size > 0
