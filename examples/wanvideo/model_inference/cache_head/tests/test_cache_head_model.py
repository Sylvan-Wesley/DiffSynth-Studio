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
    load_cache_head,
    save_cache_head,
    token_grid,
    unpatchify_tokens,
)
from fake_score_wan import FakeScoreWan, LoRALinear


# ═══════════════════════════════════════════════════════════════
# Schedule
# ═══════════════════════════════════════════════════════════════

def test_schedule_counts():
    s = CacheHeadSchedule(num_inference_steps=15, full_step_indices=(1, 2, 6, 10, 14))
    assert s.num_full_steps == 5
    assert s.num_head_steps == 10
    assert s.head_step_indices == (3, 4, 5, 7, 8, 9, 11, 12, 13, 15)


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
# CacheHead network
# ═══════════════════════════════════════════════════════════════

def test_cachehead_zero_init_is_carry_previous():
    """A freshly constructed head emits an exactly-zero residual."""
    head = CacheHead(CacheHeadConfig())
    head.eval()
    grid = (3, 2, 3)
    tokens = torch.randn(2, 3 * 2 * 3, 64)
    t = torch.tensor([500.0, 100.0])
    out = head(tokens, t, grid)
    assert out.shape == tokens.shape
    assert torch.equal(out, torch.zeros_like(out)), "zero-init output must reproduce carry_previous exactly"


def test_cachehead_forward_shapes_and_grid_mismatch():
    head = CacheHead(CacheHeadConfig())
    head.eval()
    with pytest.raises(ValueError):
        head(torch.randn(1, 60, 64), torch.tensor([500.0]), (3, 4, 4))  # 3*4*4 = 48 != 60


def test_cachehead_learns_away_from_zero():
    """Trained against a velocity target (prev + residual), the head moves the
    residual away from the zero-init carry_previous.  A loss directly on the
    residual would have zero gradient through the zero-init out_proj, so the
    test uses the real training target: ``v_hat = prev_guided + residual``."""
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

    # A single optimizer step moves the residual away from zero.
    opt = torch.optim.AdamW(head.parameters(), lr=1e-1)
    opt.zero_grad()
    v_hat = prev_guided + head(tokens, t, grid)
    v_hat.pow(2).mean().backward()
    opt.step()
    head.eval()
    with torch.no_grad():
        residual1 = head(tokens, t, grid)
    assert not torch.equal(residual1, torch.zeros_like(residual1))


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


def test_checkpoint_missing_file():
    with pytest.raises(FileNotFoundError):
        load_cache_head("/nonexistent/path/head.ckpt")
