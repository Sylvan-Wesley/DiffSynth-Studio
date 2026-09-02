"""CPU integration tests for training the sparse-attention student.

These drive the real ``CacheHeadTrainer`` supervised loop with a tiny real
``WanModel`` as the teacher, so the whole path is exercised: teacher rollout ->
trajectory cache -> prompt-embedding cache -> latent reconstruction -> sparse
student forward -> loss -> checkpoint.

Run from the repo root or the cache_head dir:
    pytest examples/wanvideo/model_inference/cache_head/tests/test_sparse_training.py
"""

import pytest
import torch

from diffsynth.models.wan_video_dit import WanModel

from cache_head_model import CacheHeadConfig, CacheHeadSchedule
from cache_head_model_inference import head_step
from cache_head_model_training import CacheHeadTrainer
from sparse_cache_head import SparseCacheHead, load_sparse_cache_head, resolve_layer_indices


PATCH_SIZE = (1, 2, 2)
GRID = (2, 3, 4)                                  # S = 24
LATENT_SHAPE = (1, 16, GRID[0], GRID[1] * 2, GRID[2] * 2)
TEXT_DIM = 8
SCHEDULE = CacheHeadSchedule(num_inference_steps=4, full_step_indices=(1, 2))


class FakeScheduler:
    """FlowMatchScheduler's surface, reduced to what the trainer touches."""

    def __init__(self, num_steps=4):
        self.num_train_timesteps = 1000.0
        self.set_timesteps(num_steps)

    def set_timesteps(self, num_steps, denoising_strength=1.0, shift=5.0):
        self.num_inference_steps = num_steps
        self.timesteps = torch.linspace(999.0, 1.0, num_steps)
        self.sigmas = self.timesteps / 1000.0

    def step(self, model_output, timestep, sample):
        return sample + 0.1 * model_output


def tiny_teacher(seed: int = 0) -> WanModel:
    torch.manual_seed(seed)
    dit = WanModel(
        dim=32, in_dim=16, ffn_dim=64, out_dim=16, text_dim=TEXT_DIM, freq_dim=16, eps=1e-6,
        patch_size=PATCH_SIZE, num_heads=4, num_layers=2, has_image_input=False,
    ).eval()
    dit._cache_head_model_id = "test/tiny-wan"
    return dit


def encode(caption):
    captions = [caption] if isinstance(caption, str) else list(caption)
    values = [sum(ord(c) for c in text) / 1000.0 for text in captions]
    return torch.stack([torch.full((4, TEXT_DIM), v) for v in values])


def make_trainer(tmp_path, *, pattern="spatiotemporal_window", num_layers=None, **overrides):
    torch.manual_seed(0)
    teacher = tiny_teacher()
    config = CacheHeadConfig(
        head_variant="sparse_dit",
        version=4,
        schedule=SCHEDULE,
        sparse_pattern=pattern,
        sparse_spatial_radius=1,
        sparse_temporal_radius=1,
        latent_channels=16,
        fusion_hidden_channels=8,
        student_layer_indices=resolve_layer_indices(
            len(teacher.blocks), num_layers=num_layers
        ),
    )
    head = SparseCacheHead(teacher, config, use_gradient_checkpointing=False)
    kwargs = dict(
        dit=teacher,
        scheduler=FakeScheduler(SCHEDULE.num_inference_steps),
        head=head,
        fake_score=None,
        text_encode=encode,
        neg_ctx=torch.zeros(1, 4, TEXT_DIM),
        dataset=[("0", "a dog runs"), ("1", "a cat sleeps")],
        schedule=SCHEDULE,
        cfg_scale=5.0,
        patch_size=PATCH_SIZE,
        grid=GRID,
        latent_shape=LATENT_SHAPE,
        arm="supervised",
        device="cpu",
        dtype=torch.float32,
        batch_size=1,
        micro_batch=1,
        epochs=1,
        optimizer_steps_per_iteration=1,
        trajectory_dir=tmp_path,
        text_encode_batch=encode,
        seed=0,
    )
    kwargs.update(overrides)
    return CacheHeadTrainer(**kwargs), head


def cached_batch(trainer, items, split="train"):
    guided = trainer.load_teacher_batch(items, split=split).to(
        device=trainer.device, dtype=trainer.dtype
    )
    latents = trainer.deterministic_initial_latents(items, split=split)
    context = trainer.load_context_batch(items)
    return guided, latents, context


# ═══════════════════════════════════════════════════════════════
# Wiring
# ═══════════════════════════════════════════════════════════════

def test_trainer_knows_the_sparse_student_needs_text(tmp_path):
    trainer, _ = make_trainer(tmp_path)
    assert trainer.needs_context


def test_token_head_does_not_request_text(tmp_path):
    from cache_head_model import CacheHead

    trainer, _ = make_trainer(
        tmp_path, head=CacheHead(CacheHeadConfig(head_variant="latent_fusion", version=3))
    )
    assert not trainer.needs_context


def test_teacher_stays_frozen_while_the_student_trains(tmp_path):
    trainer, head = make_trainer(tmp_path)
    assert not any(p.requires_grad for p in trainer.dit.parameters())
    assert all(p.requires_grad for p in head.dit.parameters())
    assert all(p.requires_grad for p in head.fusion.parameters())


def test_optimizer_covers_the_dit_and_the_fusion(tmp_path):
    """'Full DiT + conv' means the optimizer must actually see both."""
    trainer, head = make_trainer(tmp_path)
    optimized = {id(p) for group in trainer.head_opt.param_groups for p in group["params"]}
    assert all(id(p) in optimized for p in head.dit.parameters())
    assert all(id(p) in optimized for p in head.fusion.parameters())


# ═══════════════════════════════════════════════════════════════
# The supervised objective
# ═══════════════════════════════════════════════════════════════

def test_supervised_step_produces_a_finite_loss_over_the_head_steps(tmp_path):
    trainer, _ = make_trainer(tmp_path)
    items = list(trainer.dataset)[:1]
    guided, latents, context = cached_batch(trainer, items)
    out = trainer.supervised_teacher_forced(guided, latents, context)
    assert torch.isfinite(out["loss"])
    assert [rec["step"] for rec in out["per_step"]] == list(SCHEDULE.head_step_indices)
    for rec in out["per_step"]:
        assert rec["carry_mse"] > 0            # the carry baseline stays the reference
        assert "relative_improvement" in rec


def test_supervised_step_requires_the_prompt_context(tmp_path):
    trainer, _ = make_trainer(tmp_path)
    items = list(trainer.dataset)[:1]
    guided, latents, _ = cached_batch(trainer, items)
    with pytest.raises(ValueError, match="requires the positive prompt context"):
        trainer.supervised_teacher_forced(guided, latents, None)


def test_supervised_step_predicts_directly_rather_than_as_a_carry_residual(tmp_path):
    """A zero-init fusion makes the *input* identical to the teacher's, but the
    student still emits an unguided velocity, so it must not coincide with the
    carry baseline."""
    trainer, _ = make_trainer(tmp_path, pattern="dense")
    items = list(trainer.dataset)[:1]
    guided, latents, context = cached_batch(trainer, items)
    out = trainer.supervised_teacher_forced(guided, latents, context)
    for rec in out["per_step"]:
        assert rec["loss"] != pytest.approx(0.0, abs=1e-12)


def test_gradients_reach_the_student_through_the_supervised_loss(tmp_path):
    trainer, head = make_trainer(tmp_path)
    items = list(trainer.dataset)[:1]
    guided, latents, context = cached_batch(trainer, items)
    trainer.supervised_teacher_forced(guided, latents, context)["loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in head.dit.parameters()
    )
    assert head.fusion.net[-1].weight.grad.abs().sum() > 0


# ═══════════════════════════════════════════════════════════════
# The prompt-embedding cache
# ═══════════════════════════════════════════════════════════════

def test_context_cache_round_trips_and_then_hits(tmp_path):
    trainer, _ = make_trainer(tmp_path)
    items = list(trainer.dataset)
    first = trainer.load_context_batch(items)
    assert trainer.context_cache_misses == len(items)
    assert trainer.context_cache_hits == 0
    torch.testing.assert_close(first.cpu(), encode([c for _, c in items]))

    second = trainer.load_context_batch(items)
    assert trainer.context_cache_hits == len(items)
    torch.testing.assert_close(first, second)


def test_context_cache_is_keyed_by_caption_not_by_id(tmp_path):
    """Embeddings depend only on the caption and the text encoder, so they are
    shared across trajectory fingerprints and splits."""
    trainer, _ = make_trainer(tmp_path)
    a = trainer._context_path("a dog runs")
    b = trainer._context_path("a dog runs")
    c = trainer._context_path("a cat sleeps")
    assert a == b and a != c


def test_corrupted_context_cache_is_reported(tmp_path):
    trainer, _ = make_trainer(tmp_path)
    items = list(trainer.dataset)[:1]
    trainer.load_context_batch(items)
    path = trainer._context_path(items[0][1])
    torch.save({"metadata": {"caption_sha256": "wrong"}, "context": torch.zeros(4, 8)}, path)
    with pytest.raises(RuntimeError, match="does not match its caption"):
        trainer.load_context_batch(items)


def test_prefetch_warms_both_caches(tmp_path):
    trainer, _ = make_trainer(tmp_path)
    trainer.prefetch_split(trainer.dataset, split="train")
    assert trainer.trajectory_cache_misses == len(trainer.dataset)
    assert trainer.context_cache_misses == len(trainer.dataset)
    trainer.prefetch_split(trainer.dataset, split="train")
    assert trainer.trajectory_cache_hits == len(trainer.dataset)
    assert trainer.context_cache_hits == len(trainer.dataset)


# ═══════════════════════════════════════════════════════════════
# head_step dispatch
# ═══════════════════════════════════════════════════════════════

def test_head_step_routes_the_sparse_variant_and_returns_a_latent_velocity(tmp_path):
    _, head = make_trainer(tmp_path)
    head.eval()
    prev = torch.randn(1, GRID[0] * GRID[1] * GRID[2], 64)
    latents = torch.randn(*LATENT_SHAPE)
    with torch.no_grad():
        noise_pred, v_tokens = head_step(
            head, torch.tensor([500.0]), prev, GRID, PATCH_SIZE,
            current_latents=latents, context=torch.randn(1, 4, TEXT_DIM),
        )
    assert noise_pred.shape == LATENT_SHAPE
    assert v_tokens.shape == prev.shape
    # The sparse student predicts outright; it must not be a carry residual.
    assert not torch.allclose(v_tokens, prev)


def test_head_step_rejects_a_missing_context(tmp_path):
    _, head = make_trainer(tmp_path)
    prev = torch.randn(1, GRID[0] * GRID[1] * GRID[2], 64)
    with pytest.raises(ValueError, match="requires the positive prompt context"):
        head_step(
            head, torch.tensor([500.0]), prev, GRID, PATCH_SIZE,
            current_latents=torch.randn(*LATENT_SHAPE), context=None,
        )


def test_head_step_rejects_a_missing_latent(tmp_path):
    _, head = make_trainer(tmp_path)
    prev = torch.randn(1, GRID[0] * GRID[1] * GRID[2], 64)
    with pytest.raises(ValueError, match="requires the live current latent"):
        head_step(head, torch.tensor([500.0]), prev, GRID, PATCH_SIZE, current_latents=None)


# ═══════════════════════════════════════════════════════════════
# End to end
# ═══════════════════════════════════════════════════════════════

def test_full_supervised_epoch_trains_and_checkpoints(tmp_path):
    trainer, head = make_trainer(tmp_path / "cache")
    before = head.fusion.net[-1].weight.detach().clone()

    save_dir = tmp_path / "run"
    trainer.train(checkpoint_every=0, save_dir=save_dir, log_interval=1)

    assert (save_dir / "cache_head_final.ckpt").is_file()
    assert not torch.equal(head.fusion.net[-1].weight, before), "the adapter did not move"
    assert trainer.logs and all(record["finite"] for record in trainer.logs)

    restored, config = load_sparse_cache_head(
        save_dir / "cache_head_final.ckpt", tiny_teacher(seed=3)
    )
    assert config.head_variant == "sparse_dit"
    assert config.sparse_pattern == "spatiotemporal_window"
    head.eval()
    prev = torch.randn(1, GRID[0] * GRID[1] * GRID[2], 64)
    latents = torch.randn(*LATENT_SHAPE)
    context = torch.randn(1, 4, TEXT_DIM)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(prev, latents, torch.tensor([500.0]), context, GRID),
            head(prev, latents, torch.tensor([500.0]), context, GRID),
        )


def test_shallower_student_trains_end_to_end(tmp_path):
    trainer, head = make_trainer(tmp_path / "cache", num_layers=1)
    assert len(head.dit.blocks) == 1
    trainer.train(checkpoint_every=0, save_dir=tmp_path / "run", log_interval=1)
    assert trainer.logs and all(record["finite"] for record in trainer.logs)
