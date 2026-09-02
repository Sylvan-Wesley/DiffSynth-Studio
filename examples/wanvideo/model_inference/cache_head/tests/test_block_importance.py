"""CPU tests for the DiT block-importance profiler.

Run from the repo root or the cache_head dir:
    pytest examples/wanvideo/model_inference/cache_head/tests/test_block_importance.py
"""

import pytest
import torch

from diffsynth.models.wan_video_dit import WanModel

from profile_block_importance import (
    BlockContributionProbe,
    select_important_layers,
    summarize,
)


# ═══════════════════════════════════════════════════════════════
# Selection
# ═══════════════════════════════════════════════════════════════

def test_selection_keeps_the_highest_scoring_blocks():
    scores = [0.9, 0.1, 0.8, 0.2, 0.7]
    assert select_important_layers(scores, 3) == (0, 2, 4)


def test_selection_always_pins_the_endpoints():
    """The first block reads the patch embedding and the last feeds the output
    head, so both stay regardless of their measured score."""
    scores = [0.0, 0.9, 0.9, 0.9, 0.0]
    assert select_important_layers(scores, 3) == (0, 1, 4)


def test_selection_returns_everything_when_nothing_is_dropped():
    assert select_important_layers([0.3, 0.1, 0.2], 3) == (0, 1, 2)


def test_selection_of_one_block_keeps_the_first():
    assert select_important_layers([0.1, 0.9, 0.5], 1) == (0,)


def test_selection_is_sorted_ascending():
    scores = [0.5, 0.1, 0.9, 0.3, 0.7, 0.2]
    picked = select_important_layers(scores, 4)
    assert list(picked) == sorted(picked)
    assert len(set(picked)) == 4


@pytest.mark.parametrize("keep", [0, -1, 7])
def test_selection_rejects_an_impossible_count(keep):
    with pytest.raises(ValueError, match="num_layers must be in"):
        select_important_layers([0.1] * 6, keep)


def test_selection_rejects_empty_scores():
    with pytest.raises(ValueError, match="non-empty"):
        select_important_layers([], 1)


def test_selection_output_feeds_the_student_layer_indices_flag():
    from sparse_cache_head import resolve_layer_indices

    picked = select_important_layers([0.4, 0.1, 0.9, 0.2, 0.6], 3)
    assert resolve_layer_indices(5, explicit=picked) == picked


# ═══════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════

def test_summarize_averages_over_steps():
    records = [
        {"layer": 0, "step": 0, "relative_delta": 0.2, "cosine": 0.9},
        {"layer": 0, "step": 1, "relative_delta": 0.4, "cosine": 0.7},
        {"layer": 1, "step": 0, "relative_delta": 1.0, "cosine": 0.1},
    ]
    rows = summarize(records, num_layers=2)
    assert rows[0]["relative_delta"] == pytest.approx(0.3)
    assert rows[0]["cosine"] == pytest.approx(0.8)
    assert rows[0]["samples"] == 2
    assert rows[1]["relative_delta"] == pytest.approx(1.0)


def test_summarize_emits_a_row_for_a_block_that_was_never_measured():
    rows = summarize([{"layer": 0, "step": 0, "relative_delta": 0.5, "cosine": 1.0}], num_layers=3)
    assert [row["layer"] for row in rows] == [0, 1, 2]
    assert rows[2]["samples"] == 0
    assert rows[2]["relative_delta"] == 0.0


# ═══════════════════════════════════════════════════════════════
# The probe
# ═══════════════════════════════════════════════════════════════

def tiny_dit(num_layers: int = 3) -> WanModel:
    torch.manual_seed(0)
    return WanModel(
        dim=32, in_dim=16, ffn_dim=64, out_dim=16, text_dim=48, freq_dim=16, eps=1e-6,
        patch_size=(1, 2, 2), num_heads=4, num_layers=num_layers, has_image_input=False,
    ).eval()


def dit_inputs():
    torch.manual_seed(1)
    return torch.randn(1, 16, 2, 8, 8), torch.tensor([500.0]), torch.randn(1, 7, 48)


def test_probe_records_one_row_per_block_per_step():
    dit = tiny_dit()
    probe = BlockContributionProbe(dit)
    probe.install()
    try:
        x, t, ctx = dit_inputs()
        for step in range(2):
            probe.step = step
            probe.enabled = True
            with torch.no_grad():
                dit(x=x, timestep=t, context=ctx)
            probe.enabled = False
    finally:
        probe.restore()
    assert len(probe.records) == 2 * len(dit.blocks)
    assert {r["layer"] for r in probe.records} == set(range(len(dit.blocks)))
    assert {r["step"] for r in probe.records} == {0, 1}
    assert all(r["relative_delta"] >= 0 for r in probe.records)
    assert all(-1.0 <= r["cosine"] <= 1.0 for r in probe.records)


def test_probe_does_not_change_the_model_output():
    """The probe is a side channel; the trajectory it measures must be the one
    the model would have produced anyway."""
    dit = tiny_dit()
    x, t, ctx = dit_inputs()
    with torch.no_grad():
        expected = dit(x=x, timestep=t, context=ctx)
    probe = BlockContributionProbe(dit)
    probe.install()
    probe.enabled = True
    try:
        with torch.no_grad():
            actual = dit(x=x, timestep=t, context=ctx)
    finally:
        probe.restore()
    assert torch.equal(actual, expected)


def test_probe_restores_the_original_forwards():
    dit = tiny_dit()
    originals = [block.forward for block in dit.blocks]
    probe = BlockContributionProbe(dit)
    probe.install()
    assert all(block.forward is not original for block, original in zip(dit.blocks, originals))
    probe.restore()
    assert all(block.forward == original for block, original in zip(dit.blocks, originals))


def test_probe_records_nothing_while_disabled():
    dit = tiny_dit()
    probe = BlockContributionProbe(dit)
    probe.install()
    try:
        x, t, ctx = dit_inputs()
        with torch.no_grad():
            dit(x=x, timestep=t, context=ctx)
    finally:
        probe.restore()
    assert probe.records == []


def test_an_identity_block_scores_zero_contribution():
    """A block that returns its input unchanged is exactly what the profiler
    should mark as droppable."""
    dit = tiny_dit()
    probe = BlockContributionProbe(dit)
    probe.install()
    try:
        # Replace one wrapped block's underlying forward with the identity.
        dit.blocks[1].forward = lambda x, *args, **kwargs: x
        probe.install()  # re-wrap so the identity is measured
        probe.enabled = True
        x, t, ctx = dit_inputs()
        with torch.no_grad():
            dit(x=x, timestep=t, context=ctx)
    finally:
        probe.enabled = False
        probe.restore()
    identity = [r for r in probe.records if r["layer"] == 1]
    assert identity
    assert all(r["relative_delta"] == pytest.approx(0.0, abs=1e-6) for r in identity)
    assert all(r["cosine"] == pytest.approx(1.0, abs=1e-6) for r in identity)
