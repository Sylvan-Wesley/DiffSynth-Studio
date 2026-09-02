"""CPU-runnable tests for the CacheHead residual network, schedule, LoRA
fake-score estimator, and checkpoint I/O.

Run from the repo root or the cache_head dir:
    pytest examples/wanvideo/model_inference/cache_head/tests/test_cache_head_model.py
"""

import os
import tempfile

import einops
import pytest
import torch

from cache_head_model import (
    CacheHead,
    CacheHeadConfig,
    CacheHeadSchedule,
    HEAD_VARIANTS,
    load_cache_head,
    patchify_latents,
    parse_full_step_indices,
    save_cache_head,
    token_grid,
    unpatchify_tokens,
)
from fake_score_wan import FakeScoreWan, LoRALinear


# ═══════════════════════════════════════════════════════════════
# Schedule
# ═══════════════════════════════════════════════════════════════

def test_schedule_counts():
    s = CacheHeadSchedule()
    assert s.num_full_steps == 7
    assert s.num_head_steps == 8
    assert s.full_step_indices == (1, 2, 3, 4, 5, 6, 7)
    assert s.head_step_indices == (8, 9, 10, 11, 12, 13, 14, 15)


def test_schedule_0indexed_mapping():
    s = CacheHeadSchedule(15, (1, 2, 6, 10, 14))
    assert [i for i in range(15) if s.is_full_step(i)] == [0, 1, 5, 9, 13]
    assert [i for i in range(15) if s.is_head_step(i)] == [2, 3, 4, 6, 7, 8, 10, 11, 12, 14]


@pytest.mark.parametrize("bad", [(1, 1, 6, 10, 14), (0, 2), (1, 2, 16), (1, 5, 2)])
def test_schedule_rejects_bad_indices(bad):
    with pytest.raises(ValueError):
        CacheHeadSchedule(15, bad)


def test_schedule_rejects_empty():
    with pytest.raises(ValueError):
        CacheHeadSchedule(15, ())


# ═══════════════════════════════════════════════════════════════
# --full-steps CLI parsing (shared by the training and inference CLIs)
# ═══════════════════════════════════════════════════════════════

def test_parse_full_step_indices_basic():
    assert parse_full_step_indices("1,2,6,10,14") == (1, 2, 6, 10, 14)


def test_parse_full_step_indices_strips_whitespace():
    assert parse_full_step_indices(" 1, 2 ,6 ") == (1, 2, 6)


def test_parse_full_step_indices_tolerates_stray_commas():
    # Trailing/doubled commas are a harmless typo, not a parse error.
    assert parse_full_step_indices("1,2,6,") == (1, 2, 6)
    assert parse_full_step_indices("1,,3") == (1, 3)


@pytest.mark.parametrize("bad", ["", "  ", "1,a,3"])
def test_parse_full_step_indices_rejects_garbage(bad):
    with pytest.raises(ValueError):
        parse_full_step_indices(bad)


def test_parse_full_step_indices_feeds_schedule_validation():
    # A syntactically fine but semantically bad spec (duplicate, unsorted) is
    # still rejected -- by CacheHeadSchedule, not the parser itself.
    with pytest.raises(ValueError):
        CacheHeadSchedule(15, parse_full_step_indices("2,1"))


# ═══════════════════════════════════════════════════════════════
# CacheHead network
# ═══════════════════════════════════════════════════════════════

def test_cachehead_default_zero_init_is_carry_previous():
    """Every freshly constructed production head emits exactly zero."""
    head = CacheHead(CacheHeadConfig())
    head.eval()
    grid = (3, 2, 3)
    tokens = torch.randn(2, 3 * 2 * 3, 64)
    t = torch.tensor([500.0, 100.0])
    out = head(tokens, t, grid)
    assert out.shape == tokens.shape
    assert torch.equal(out, torch.zeros_like(out))


def test_cachehead_random_init_remains_an_explicit_diagnostic_option():
    head = CacheHead(CacheHeadConfig(), zero_init_out_proj=False).eval()
    tokens = torch.randn(2, 18, 64)
    out = head(tokens, torch.tensor([500.0, 100.0]), (3, 2, 3))
    assert not torch.equal(out, torch.zeros_like(out))


def test_cachehead_forward_shapes_and_grid_mismatch():
    head = CacheHead(CacheHeadConfig())
    head.eval()
    with pytest.raises(ValueError):
        head(torch.randn(1, 60, 64), torch.tensor([500.0]), (3, 4, 4))  # 3*4*4 = 48 != 60


def test_cachehead_zero_init_learns_away_from_carry_previous():
    """The zero output projection receives a gradient and changes after one update."""
    head = CacheHead(CacheHeadConfig())
    head.train()
    grid = (3, 2, 3)
    tokens = torch.randn(2, 3 * 2 * 3, 64)
    prev_guided = torch.randn(2, 3 * 2 * 3, 64)
    t = torch.tensor([500.0, 100.0])

    # Gradients reach every parameter through the velocity target.
    v_hat = prev_guided + head(tokens, t, grid)
    v_hat.pow(2).mean().backward()
    n_params = len(list(head.parameters()))
    n_grads = sum(p.grad is not None for p in head.parameters())
    assert n_grads == n_params

    # A single optimizer step changes the residual.
    with torch.no_grad():
        residual0 = head(tokens, t, grid).clone()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-1)
    opt.zero_grad()
    v_hat = prev_guided + head(tokens, t, grid)
    v_hat.pow(2).mean().backward()
    opt.step()
    head.eval()
    with torch.no_grad():
        residual1 = head(tokens, t, grid)
    assert not torch.equal(residual1, residual0)


# ═══════════════════════════════════════════════════════════════
# Token <-> velocity helpers
# ═══════════════════════════════════════════════════════════════

def test_token_grid_and_unpatchify_round_trip():
    f, h, w = 3, 4, 6  # even spatial dims (Wan latents are even)
    latent = torch.randn(2, 16, f, h, w)
    grid = token_grid(f, h, w, (1, 2, 2))
    assert grid == (3, 2, 3)

    # Identity-sparse patchify conv: token k = (y*2 + z)*16 + c reads latent[c, y, z].
    patch = torch.nn.Conv3d(16, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False)
    with torch.no_grad():
        patch.weight.zero_()
        for y in range(2):
            for z in range(2):
                for c in range(16):
                    patch.weight[(y * 2 + z) * 16 + c, c, 0, y, z] = 1.0
        tokens = patch(latent)
    tokens = einops.rearrange(tokens, "b d f h w -> b (f h w) d")
    back = unpatchify_tokens(tokens, grid, (1, 2, 2))
    assert back.shape == latent.shape
    assert torch.equal(back, latent)


def test_patchify_latents_is_exact_unpatchify_inverse():
    latent = torch.randn(2, 16, 3, 8, 12)
    grid = (3, 4, 6)
    tokens = patchify_latents(latent, grid, (1, 2, 2))
    assert tokens.shape == (2, 3 * 4 * 6, 64)
    assert torch.equal(unpatchify_tokens(tokens, grid, (1, 2, 2)), latent)


def test_patchify_unpatchify_is_exact_in_both_directions():
    tokens = torch.randn(2, 24, 64)
    latent = unpatchify_tokens(tokens, (2, 3, 4), (1, 2, 2))
    assert torch.equal(patchify_latents(latent, (2, 3, 4), (1, 2, 2)), tokens)


# ═══════════════════════════════════════════════════════════════
# LoRA fake-score estimator
# ═══════════════════════════════════════════════════════════════

class _DummyDit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Linear(16, 8))

    def forward(self, x):
        return self.net(x)


def test_lora_delta_zero_at_init():
    dummy = _DummyDit()
    fake = FakeScoreWan(dummy, rank=4, alpha=1.0)
    x = torch.randn(5, 8)
    with torch.no_grad():
        assert torch.allclose(dummy(x), fake(x), atol=0.0)


def test_lora_only_ab_trainable():
    dummy = _DummyDit()
    fake = FakeScoreWan(dummy, rank=4, alpha=1.0)
    trainable = [n for n, p in fake.named_parameters() if p.requires_grad]
    assert len(trainable) == 4
    assert all(".lora_A" in n or ".lora_B" in n for n in trainable)
    # Base weights inside the clone are frozen.
    for n, p in fake.base.named_parameters():
        if ".lora_A" not in n and ".lora_B" not in n:
            assert not p.requires_grad


def test_lora_gradients_flow_only_to_ab():
    dummy = _DummyDit()
    fake = FakeScoreWan(dummy, rank=4, alpha=1.0)
    fake.train()
    x = torch.randn(5, 8)
    fake(x).pow(2).mean().backward()
    for n, p in fake.named_parameters():
        if ".lora_A" in n or ".lora_B" in n:
            assert p.grad is not None
        else:
            assert p.grad is None


# ═══════════════════════════════════════════════════════════════
# Checkpoint I/O
# ═══════════════════════════════════════════════════════════════

def test_checkpoint_round_trip():
    cfg = CacheHeadConfig()
    head = CacheHead(cfg)
    head.eval()
    tmp = tempfile.mkdtemp()
    path = save_cache_head(head, cfg, os.path.join(tmp, "head.ckpt"))
    head2, cfg2 = load_cache_head(path)
    assert cfg2 == cfg
    assert all(
        torch.equal(a, b)
        for a, b in zip(head.state_dict().values(), head2.state_dict().values())
    )


@pytest.mark.parametrize("variant", HEAD_VARIANTS[1:])
def test_version_three_variant_checkpoint_round_trip(tmp_path, variant):
    cfg = CacheHeadConfig(head_variant=variant, version=3)
    head = CacheHead(cfg, zero_init_out_proj=False).eval()
    path = save_cache_head(head, cfg, tmp_path / f"{variant}.ckpt")
    payload = torch.load(path, weights_only=False)
    loaded, loaded_cfg = load_cache_head(path)
    assert payload["version"] == 3
    assert loaded_cfg == cfg
    assert loaded_cfg.head_variant == variant
    assert all(
        torch.equal(a, b)
        for a, b in zip(head.state_dict().values(), loaded.state_dict().values())
    )


def test_version_two_checkpoint_without_variant_remains_bit_identical(tmp_path):
    cfg = CacheHeadConfig()
    head = CacheHead(cfg, zero_init_out_proj=False).eval()
    tokens = torch.randn(1, 24, 64)
    timestep = torch.tensor([500.0])
    expected = head(tokens, timestep, (2, 3, 4))
    path = save_cache_head(head, cfg, tmp_path / "v2.ckpt")
    payload = torch.load(path, weights_only=False)
    payload["config"].pop("head_variant")
    torch.save(payload, path)

    loaded, loaded_cfg = load_cache_head(path)
    assert loaded_cfg.head_variant == "legacy"
    assert torch.equal(loaded(tokens, timestep, (2, 3, 4)), expected)


def test_checkpoint_missing_file():
    with pytest.raises(FileNotFoundError):
        load_cache_head("/nonexistent/path/head.ckpt")


# ═══════════════════════════════════════════════════════════════
# Mixed precision
# ═══════════════════════════════════════════════════════════════

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_head_forward_runs_in_every_precision(dtype):
    """The timestep embedding is built in float32 but the AdaLN projection
    follows the model dtype; without a cast F.linear raises
    "mat1 and mat2 must have the same dtype" on any --precision bf16 run.
    """
    head = CacheHead(CacheHeadConfig()).to(dtype=dtype)
    tokens = torch.randn(2, 24, 64, dtype=dtype)
    timestep = torch.tensor([999.0], dtype=dtype)
    out = head(tokens, timestep, (2, 3, 4))
    assert out.dtype == dtype
    assert out.shape == tokens.shape
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_head_backward_runs_in_low_precision(dtype):
    head = CacheHead(CacheHeadConfig()).to(dtype=dtype)
    tokens = torch.randn(1, 24, 64, dtype=dtype)
    out = head(tokens, torch.tensor([500.0], dtype=dtype), (2, 3, 4))
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in head.parameters())


def test_head_accepts_a_float32_timestep_in_a_bf16_model():
    """Timesteps reach the head straight from the scheduler, which builds them
    in float32 regardless of --precision."""
    head = CacheHead(CacheHeadConfig()).to(dtype=torch.bfloat16)
    tokens = torch.randn(1, 24, 64, dtype=torch.bfloat16)
    out = head(tokens, torch.tensor([999.0], dtype=torch.float32), (2, 3, 4))
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()


@pytest.mark.parametrize("variant", HEAD_VARIANTS[1:])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_latent_variants_forward_backward_in_every_precision(variant, dtype):
    cfg = CacheHeadConfig(head_variant=variant, version=3)
    head = CacheHead(cfg).to(dtype=dtype)
    tokens = torch.randn(1, 24, 64, dtype=dtype)
    latent_tokens = torch.randn_like(tokens)
    out = head(
        tokens, torch.tensor([500.0], dtype=dtype), (2, 3, 4),
        latent_tokens=latent_tokens,
    )
    assert out.shape == tokens.shape
    assert out.dtype == dtype
    assert torch.isfinite(out.float()).all()
    out.float().pow(2).mean().backward()
    assert all(p.grad is not None for p in head.parameters())


@pytest.mark.parametrize("variant", HEAD_VARIANTS[1:])
def test_latent_variants_require_exact_latent_token_shape(variant):
    head = CacheHead(CacheHeadConfig(head_variant=variant, version=3))
    tokens = torch.randn(1, 24, 64)
    with pytest.raises(ValueError, match="requires latent_tokens"):
        head(tokens, torch.tensor([500.0]), (2, 3, 4))
    with pytest.raises(ValueError, match="must match"):
        head(tokens, torch.tensor([500.0]), (2, 3, 4), torch.randn(1, 23, 64))


@pytest.mark.parametrize("variant", HEAD_VARIANTS[1:])
def test_zero_init_latent_variants_are_exact_carry(variant):
    head = CacheHead(CacheHeadConfig(head_variant=variant, version=3))
    tokens = torch.randn(1, 24, 64)
    residual = head(
        tokens, torch.tensor([500.0]), (2, 3, 4),
        latent_tokens=torch.randn_like(tokens),
    )
    assert torch.equal(residual, torch.zeros_like(tokens))


def test_residual_variants_have_expected_receptive_field_depth_and_modulation_init():
    one = CacheHead(CacheHeadConfig(head_variant="latent_residual", version=3))
    two = CacheHead(CacheHeadConfig(head_variant="latent_residual_deep", version=3))
    assert len(one.blocks) == 1
    assert len(two.blocks) == 2

    for block in list(one.blocks) + list(two.blocks):
        for modulation in (block.mlp_adaln, block.mixer_adaln):
            weight = modulation.net[-1].weight
            scale_bias, shift_bias, gate_bias = modulation.net[-1].bias.chunk(3)
            assert torch.count_nonzero(weight) == 0
            assert torch.equal(scale_bias, torch.zeros_like(scale_bias))
            assert torch.equal(shift_bias, torch.zeros_like(shift_bias))
            assert torch.equal(gate_bias, torch.ones_like(gate_bias))


def test_checkpoint_round_trip_preserves_dtype():
    """load_cache_head must hand back a head in the pipeline's dtype.

    CacheHead is built in float32 and load_state_dict copies into those
    parameters, so without an explicit dtype a bf16 checkpoint returns float32,
    the head emits float32 tokens, the scheduler promotes the latents, and the
    next full step feeds float32 activations to a bf16 Wan.
    """
    import tempfile

    cfg = CacheHeadConfig()
    head = CacheHead(cfg).to(dtype=torch.bfloat16)
    path = os.path.join(tempfile.mkdtemp(), "head.ckpt")
    save_cache_head(head, cfg, path)

    stored = torch.load(path, weights_only=False)["model_state_dict"]
    assert stored["out_proj.weight"].dtype == torch.bfloat16

    loaded, _ = load_cache_head(path, dtype=torch.bfloat16)
    assert next(loaded.parameters()).dtype == torch.bfloat16
    tokens = torch.randn(1, 24, 64, dtype=torch.bfloat16)
    assert loaded(tokens, torch.tensor([500.0]), (2, 3, 4)).dtype == torch.bfloat16


def test_version_one_checkpoint_preserves_legacy_residual_scale(tmp_path):
    cfg = CacheHeadConfig()
    head = CacheHead(cfg, zero_init_out_proj=False).eval()
    path = tmp_path / "legacy.ckpt"
    save_cache_head(head, cfg, path)
    payload = torch.load(path, weights_only=False)
    payload["version"] = 1
    payload["config"].pop("residual_scale")
    payload["config"]["version"] = 1
    torch.save(payload, path)

    loaded, loaded_cfg = load_cache_head(path)
    tokens = torch.randn(1, 24, 64)
    timestep = torch.tensor([500.0])
    expected = head(tokens, timestep, (2, 3, 4)) * 0.1
    assert loaded_cfg.residual_scale == pytest.approx(0.1)
    assert torch.allclose(loaded(tokens, timestep, (2, 3, 4)), expected)
