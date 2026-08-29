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
    head = CacheHead(CacheHeadConfig())          # zero-init out_proj
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


# ═══════════════════════════════════════════════════════════════
# Rollout video
# ═══════════════════════════════════════════════════════════════

def test_collect_returns_final_latents_for_decoding():
    _, _, _, result = _collect(n_batch=2)
    assert result["final_latents"].shape == (2, *LATENT_SHAPE[1:])


def test_saved_arrays_exclude_the_latents(tmp_path):
    _, _, _, result = _collect()
    pt = save_error_arrays(result, tmp_path / "e.pt")
    reloaded = torch.load(pt, weights_only=False)
    # Large, device-resident, and already represented by the video.
    assert "final_latents" not in reloaded
    assert "errors" in reloaded and "timesteps" in reloaded


class _FakePipe:
    """Minimal stand-in for WanVideoPipeline's decode surface."""

    class _VAE:
        def decode(self, latents, device=None, tiled=None, tile_size=None, tile_stride=None):
            assert latents.shape[0] == 1, "decode one prompt at a time"
            return latents

    def __init__(self):
        self.vae = self._VAE()

    def vae_output_to_video(self, video):
        return video


def _stub_save_video(monkeypatch):
    """Stub diffsynth.utils.data.save_video; it is imported inside the function."""
    import sys, types

    written = []
    mod = types.ModuleType("diffsynth.utils.data")
    mod.save_video = lambda frames, path, fps=15, quality=5: (
        written.append(path), open(path, "wb").write(b"MP4")
    )
    monkeypatch.setitem(sys.modules, "diffsynth", types.ModuleType("diffsynth"))
    monkeypatch.setitem(sys.modules, "diffsynth.utils", types.ModuleType("diffsynth.utils"))
    monkeypatch.setitem(sys.modules, "diffsynth.utils.data", mod)
    return written


def test_single_prompt_keeps_the_requested_filename(tmp_path, monkeypatch):
    from cache_head_error_heatmap import save_rollout_video

    written = _stub_save_video(monkeypatch)
    latents = torch.randn(1, *LATENT_SHAPE[1:])
    out = save_rollout_video(_FakePipe(), latents, tmp_path / "rollout.mp4", device="cpu")

    assert [p.name for p in out] == ["rollout.mp4"]
    assert len(written) == 1
    assert (tmp_path / "rollout.mp4").is_file()


def test_batch_gets_one_indexed_file_per_prompt(tmp_path, monkeypatch):
    from cache_head_error_heatmap import save_rollout_video

    _stub_save_video(monkeypatch)
    latents = torch.randn(3, *LATENT_SHAPE[1:])
    out = save_rollout_video(_FakePipe(), latents, tmp_path / "rollout.mp4", device="cpu")

    assert [p.name for p in out] == ["rollout-0.mp4", "rollout-1.mp4", "rollout-2.mp4"]
    assert all(p.is_file() for p in out)


def test_video_comes_from_the_same_rollout_as_the_heatmap(tmp_path, monkeypatch):
    """The point of decoding here: the video and the panels describe one run."""
    from cache_head_error_heatmap import save_rollout_video

    _stub_save_video(monkeypatch)
    _, _, _, result = _collect(n_batch=1)
    render_error_heatmap(result, tmp_path / "heat.png")
    out = save_rollout_video(_FakePipe(), result["final_latents"], tmp_path / "r.mp4",
                             device="cpu")
    assert (tmp_path / "heat.png").is_file() and out[0].is_file()
