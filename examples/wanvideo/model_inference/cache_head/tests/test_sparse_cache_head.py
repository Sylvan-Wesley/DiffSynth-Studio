"""CPU-runnable tests for the sparse-attention CacheHead student.

The load-bearing test here is :func:`test_dense_student_reproduces_the_teacher_exactly`:
it proves weight inheritance, the zero-init conv fusion, and the attention swap
are all correct simultaneously.

Run from the repo root or the cache_head dir:
    pytest examples/wanvideo/model_inference/cache_head/tests/test_sparse_cache_head.py
"""

import tempfile
from pathlib import Path

import pytest
import torch

from diffsynth.models.wan_video_dit import WanModel

from cache_head_model import (
    CacheHead,
    CacheHeadConfig,
    CacheHeadSchedule,
    load_cache_head,
    unpatchify_tokens,
)
from sparse_attention import SparseAttentionModule
from sparse_cache_head import (
    LatentFusionConv3d,
    SparseCacheHead,
    load_sparse_cache_head,
    resolve_layer_indices,
    save_sparse_cache_head,
)


GRID = (2, 4, 4)                       # 32 tokens
LATENT_SHAPE = (1, 16, 2, 8, 8)        # [B, 16, f, 2h, 2w] with patch_size (1, 2, 2)
NUM_TOKENS = GRID[0] * GRID[1] * GRID[2]
TEXT_DIM = 48


def tiny_teacher(num_layers: int = 4, seed: int = 0) -> WanModel:
    torch.manual_seed(seed)
    return WanModel(
        dim=32, in_dim=16, ffn_dim=64, out_dim=16, text_dim=TEXT_DIM, freq_dim=16, eps=1e-6,
        patch_size=(1, 2, 2), num_heads=4, num_layers=num_layers, has_image_input=False,
    ).eval()


def sparse_config(**overrides) -> CacheHeadConfig:
    base = dict(
        head_variant="sparse_dit",
        version=4,
        schedule=CacheHeadSchedule(num_inference_steps=15),
    )
    base.update(overrides)
    return CacheHeadConfig(**base)


def sample_inputs(seed: int = 1):
    torch.manual_seed(seed)
    return (
        torch.randn(1, NUM_TOKENS, 64),       # prev_guided_tokens
        torch.randn(*LATENT_SHAPE),           # current_latents
        torch.tensor([500.0]),                # timestep
        torch.randn(1, 7, TEXT_DIM),          # positive context
    )


# ═══════════════════════════════════════════════════════════════
# Depth selection
# ═══════════════════════════════════════════════════════════════

def test_full_depth_is_the_identity_selection():
    assert resolve_layer_indices(30) == tuple(range(30))
    assert resolve_layer_indices(30, num_layers=30) == tuple(range(30))


def test_stride_selection_keeps_the_first_and_last_block():
    for k in (2, 3, 8, 15, 29):
        indices = resolve_layer_indices(30, num_layers=k)
        assert len(indices) == k
        assert indices[0] == 0
        assert indices[-1] == 29
        assert list(indices) == sorted(set(indices))


def test_single_layer_selection_keeps_the_first_block():
    assert resolve_layer_indices(30, num_layers=1) == (0,)


@pytest.mark.parametrize("num_layers", [0, -1, 31])
def test_out_of_range_depth_is_rejected(num_layers):
    with pytest.raises(ValueError, match="num_layers must be in"):
        resolve_layer_indices(30, num_layers=num_layers)


def test_explicit_indices_win_over_the_stride():
    assert resolve_layer_indices(30, num_layers=15, explicit=(0, 5, 29)) == (0, 5, 29)


@pytest.mark.parametrize(
    "explicit, message",
    [
        ((), "non-empty"),
        ((0, 0, 5), "unique"),
        ((5, 0), "sorted ascending"),
        ((0, 30), "out of range"),
        ((-1, 3), "out of range"),
    ],
)
def test_malformed_explicit_indices_are_rejected(explicit, message):
    with pytest.raises(ValueError, match=message):
        resolve_layer_indices(30, explicit=explicit)


# ═══════════════════════════════════════════════════════════════
# The conv3d fusion adapter
# ═══════════════════════════════════════════════════════════════

def test_fusion_is_the_identity_on_the_current_latents_at_init():
    """Zero-init on the last conv is what removes the need for a warm-up phase:
    the DiT's first-ever input is exactly the latent it was trained on."""
    torch.manual_seed(0)
    fusion = LatentFusionConv3d()
    prev = torch.randn(*LATENT_SHAPE)
    current = torch.randn(*LATENT_SHAPE)
    assert torch.equal(fusion(prev, current), current)


def test_fusion_preserves_the_latent_channel_count():
    """The output must be dit.in_dim wide so the inherited patch_embedding is
    fed its native input."""
    fusion = LatentFusionConv3d(latent_channels=16, hidden_channels=32)
    out = fusion(torch.randn(*LATENT_SHAPE), torch.randn(*LATENT_SHAPE))
    assert out.shape == LATENT_SHAPE


def test_fusion_stops_being_the_identity_once_trained():
    torch.manual_seed(0)
    fusion = LatentFusionConv3d()
    torch.nn.init.normal_(fusion.net[-1].weight, std=0.1)
    prev = torch.randn(*LATENT_SHAPE)
    current = torch.randn(*LATENT_SHAPE)
    assert not torch.equal(fusion(prev, current), current)


def test_fusion_actually_reads_the_previous_guided_input():
    torch.manual_seed(0)
    fusion = LatentFusionConv3d()
    torch.nn.init.normal_(fusion.net[-1].weight, std=0.1)
    current = torch.randn(*LATENT_SHAPE)
    a = fusion(torch.randn(*LATENT_SHAPE), current)
    b = fusion(torch.randn(*LATENT_SHAPE), current)
    assert not torch.allclose(a, b)


def test_fusion_rejects_mismatched_inputs():
    fusion = LatentFusionConv3d()
    with pytest.raises(ValueError, match="fusion inputs must match"):
        fusion(torch.randn(1, 16, 2, 8, 8), torch.randn(1, 16, 2, 4, 4))


@pytest.mark.parametrize("kernel_size", [0, 2, 4])
def test_fusion_requires_an_odd_kernel(kernel_size):
    with pytest.raises(ValueError, match="kernel_size must be a positive odd int"):
        LatentFusionConv3d(kernel_size=kernel_size)


# ═══════════════════════════════════════════════════════════════
# Weight inheritance
# ═══════════════════════════════════════════════════════════════

def test_student_inherits_every_teacher_weight():
    teacher = tiny_teacher()
    student = SparseCacheHead(teacher, sparse_config(), use_gradient_checkpointing=False)
    teacher_sd, student_sd = teacher.state_dict(), student.dit.state_dict()
    assert set(teacher_sd) == set(student_sd)
    assert all(torch.equal(teacher_sd[k], student_sd[k]) for k in teacher_sd)


def test_building_the_student_does_not_mutate_the_teacher():
    teacher = tiny_teacher()
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    SparseCacheHead(
        teacher,
        sparse_config(sparse_pattern="spatiotemporal_window", student_layer_indices=(0, 2)),
        use_gradient_checkpointing=False,
    )
    assert len(teacher.blocks) == 4
    assert all(isinstance(b.self_attn.attn, torch.nn.Module) for b in teacher.blocks)
    assert not any(isinstance(b.self_attn.attn, SparseAttentionModule) for b in teacher.blocks)
    after = teacher.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_student_is_trainable_even_though_the_teacher_is_frozen():
    """run_training freezes the teacher before cloning, and deepcopy carries
    requires_grad across -- so the student must re-enable it or train nothing."""
    teacher = tiny_teacher()
    teacher.requires_grad_(False)
    student = SparseCacheHead(teacher, sparse_config(), use_gradient_checkpointing=False)
    assert all(p.requires_grad for p in student.dit.parameters())
    assert all(p.requires_grad for p in student.fusion.parameters())
    assert not any(p.requires_grad for p in teacher.parameters())


def test_dense_student_reproduces_the_teacher_exactly():
    """The single most important guarantee: with the dense pattern and a
    zero-init fusion, the student IS the teacher."""
    teacher = tiny_teacher()
    student = SparseCacheHead(teacher, sparse_config(), use_gradient_checkpointing=False).eval()
    prev, latents, timestep, context = sample_inputs()
    with torch.no_grad():
        _, expected = teacher(x=latents, timestep=timestep, context=context, return_noise_tokens=True)
        actual = student(prev, latents, timestep, context, GRID)
    assert torch.equal(actual, expected)


def test_sparse_student_diverges_from_the_teacher():
    teacher = tiny_teacher()
    student = SparseCacheHead(
        teacher,
        sparse_config(sparse_pattern="spatiotemporal_window", sparse_spatial_radius=1,
                      sparse_temporal_radius=0),
        use_gradient_checkpointing=False,
    ).eval()
    prev, latents, timestep, context = sample_inputs()
    with torch.no_grad():
        _, expected = teacher(x=latents, timestep=timestep, context=context, return_noise_tokens=True)
        actual = student(prev, latents, timestep, context, GRID)
    assert not torch.allclose(actual, expected, atol=1e-4)


def test_kept_blocks_carry_their_source_teacher_weights():
    teacher = tiny_teacher(num_layers=6)
    indices = resolve_layer_indices(6, num_layers=3)
    student = SparseCacheHead(
        teacher, sparse_config(student_layer_indices=indices), use_gradient_checkpointing=False
    )
    assert student.layer_indices == indices
    assert len(student.dit.blocks) == 3
    for position, source in enumerate(indices):
        assert torch.equal(
            student.dit.blocks[position].self_attn.q.weight,
            teacher.blocks[source].self_attn.q.weight,
        )
        assert torch.equal(
            student.dit.blocks[position].modulation, teacher.blocks[source].modulation
        )


def test_every_self_attention_is_sparse_and_cross_attention_is_not():
    teacher = tiny_teacher()
    student = SparseCacheHead(
        teacher, sparse_config(sparse_pattern="spatiotemporal_window"),
        use_gradient_checkpointing=False,
    )
    assert all(isinstance(b.self_attn.attn, SparseAttentionModule) for b in student.dit.blocks)
    assert not any(isinstance(b.cross_attn.attn, SparseAttentionModule) for b in student.dit.blocks)


def test_parameter_summary_reports_the_architecture():
    teacher = tiny_teacher(num_layers=6)
    student = SparseCacheHead(
        teacher,
        sparse_config(student_layer_indices=(0, 3, 5), sparse_pattern="spatiotemporal_window"),
        use_gradient_checkpointing=False,
    )
    summary = student.parameter_summary()
    assert summary["num_layers"] == 3
    assert summary["num_teacher_layers"] == 6
    assert summary["layer_indices"] == (0, 3, 5)
    assert summary["sparse_pattern"] == "spatiotemporal_window"
    assert summary["fusion_parameters"] > 0
    assert summary["total_parameters"] == summary["dit_parameters"] + summary["fusion_parameters"]


# ═══════════════════════════════════════════════════════════════
# Forward / backward
# ═══════════════════════════════════════════════════════════════

def test_forward_returns_guided_velocity_tokens():
    student = SparseCacheHead(
        tiny_teacher(), sparse_config(sparse_pattern="spatiotemporal_window"),
        use_gradient_checkpointing=False,
    ).eval()
    prev, latents, timestep, context = sample_inputs()
    out = student(prev, latents, timestep, context, GRID)
    assert out.shape == (1, NUM_TOKENS, 64)
    # Unpatchifying the prediction must land back on the latent shape, since the
    # sampler feeds it straight to FlowMatchScheduler.step.
    assert unpatchify_tokens(out, GRID, student.patch_size).shape == LATENT_SHAPE


def test_gradients_reach_the_whole_dit_and_the_fusion():
    student = SparseCacheHead(
        tiny_teacher(), sparse_config(sparse_pattern="spatiotemporal_window"),
        use_gradient_checkpointing=False,
    )
    student.train()
    prev, latents, timestep, context = sample_inputs()
    student(prev, latents, timestep, context, GRID).pow(2).mean().backward()

    dit_params = list(student.dit.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in dit_params)
    assert any(p.grad.abs().sum() > 0 for p in dit_params)
    for param in student.fusion.parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all()


def test_fusion_input_conv_activates_once_the_output_conv_leaves_zero():
    """At init the zero-init output conv gates the branch, so the *input* conv
    legitimately sees zero gradient.  It must start learning as soon as the
    output conv moves -- otherwise the adapter would be permanently dead."""
    student = SparseCacheHead(
        tiny_teacher(), sparse_config(sparse_pattern="spatiotemporal_window"),
        use_gradient_checkpointing=False,
    )
    student.train()
    prev, latents, timestep, context = sample_inputs()

    student(prev, latents, timestep, context, GRID).pow(2).mean().backward()
    assert student.fusion.net[0].weight.grad.abs().sum() == 0
    assert student.fusion.net[-1].weight.grad.abs().sum() > 0

    optimizer = torch.optim.SGD(student.fusion.parameters(), lr=0.1)
    optimizer.step()
    student.zero_grad(set_to_none=True)

    student(prev, latents, timestep, context, GRID).pow(2).mean().backward()
    assert student.fusion.net[0].weight.grad.abs().sum() > 0


def test_gradient_checkpointing_matches_the_plain_forward():
    teacher = tiny_teacher()
    config = sparse_config(sparse_pattern="spatiotemporal_window")
    plain = SparseCacheHead(teacher, config, use_gradient_checkpointing=False).train()
    checkpointed = SparseCacheHead(teacher, config, use_gradient_checkpointing=True).train()
    checkpointed.load_state_dict(plain.state_dict())
    prev, latents, timestep, context = sample_inputs()
    torch.testing.assert_close(
        plain(prev, latents, timestep, context, GRID),
        checkpointed(prev, latents, timestep, context, GRID),
    )


# ═══════════════════════════════════════════════════════════════
# Config and checkpoint I/O
# ═══════════════════════════════════════════════════════════════

def test_sparse_variant_requires_checkpoint_version_four():
    with pytest.raises(ValueError, match="requires checkpoint version 4"):
        CacheHeadConfig(head_variant="sparse_dit", version=3)


def test_cache_head_refuses_to_build_the_sparse_variant():
    with pytest.raises(ValueError, match="not a CacheHead"):
        CacheHead(sparse_config())


def test_sparse_cache_head_refuses_a_non_sparse_config():
    with pytest.raises(ValueError, match="requires head_variant 'sparse_dit'"):
        SparseCacheHead(tiny_teacher(), CacheHeadConfig())


def test_config_normalizes_layer_indices_to_a_tuple():
    config = sparse_config(student_layer_indices=[0, 2, 3])
    assert config.student_layer_indices == (0, 2, 3)


@pytest.mark.parametrize(
    "indices, message",
    [((), "non-empty"), ((0, 0), "unique"), ((3, 1), "sorted ascending"), ((-1, 2), "non-negative")],
)
def test_config_validates_layer_indices(indices, message):
    with pytest.raises(ValueError, match=message):
        sparse_config(student_layer_indices=indices)


def test_checkpoint_round_trip_rebuilds_the_architecture_and_outputs():
    teacher = tiny_teacher(num_layers=6)
    config = sparse_config(
        sparse_pattern="spatiotemporal_window",
        sparse_spatial_radius=1,
        sparse_temporal_radius=0,
        student_layer_indices=(0, 3, 5),
        fusion_hidden_channels=24,
    )
    student = SparseCacheHead(teacher, config, use_gradient_checkpointing=False)
    # Move off the zero-init so the round-trip is not trivially satisfied.
    torch.nn.init.normal_(student.fusion.net[-1].weight, std=0.05)
    student.eval()

    prev, latents, timestep, context = sample_inputs()
    with torch.no_grad():
        expected = student(prev, latents, timestep, context, GRID)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sparse.ckpt"
        save_sparse_cache_head(student, config, path)
        # A differently-seeded teacher proves the checkpoint carries the weights
        # rather than silently reusing whatever the teacher happened to hold.
        restored, restored_config = load_sparse_cache_head(path, tiny_teacher(num_layers=6, seed=7))

    assert restored_config.student_layer_indices == (0, 3, 5)
    assert restored_config.sparse_pattern == "spatiotemporal_window"
    assert restored_config.sparse_spatial_radius == 1
    assert restored_config.fusion_hidden_channels == 24
    assert restored.layer_indices == (0, 3, 5)
    with torch.no_grad():
        torch.testing.assert_close(restored(prev, latents, timestep, context, GRID), expected)


def test_load_cache_head_points_at_the_sparse_loader():
    teacher = tiny_teacher()
    config = sparse_config()
    student = SparseCacheHead(teacher, config, use_gradient_checkpointing=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sparse.ckpt"
        save_sparse_cache_head(student, config, path)
        with pytest.raises(ValueError, match="load_sparse_cache_head"):
            load_cache_head(path)


def test_sparse_loader_rejects_a_legacy_checkpoint():
    from cache_head_model import save_cache_head

    config = CacheHeadConfig()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy.ckpt"
        save_cache_head(CacheHead(config), config, path)
        with pytest.raises(ValueError, match="not a sparse_dit checkpoint"):
            load_sparse_cache_head(path, tiny_teacher())
