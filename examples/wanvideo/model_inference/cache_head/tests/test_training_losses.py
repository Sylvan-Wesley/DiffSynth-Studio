"""CPU tests for the CacheHead training losses and gradient routing.

Uses tiny fake dit/scheduler/head so no Wan model or GPU is needed.
"""

import einops
import pytest
import torch
import torch.nn.functional as F

from cache_head_model import CacheHead, CacheHeadConfig, CacheHeadSchedule, unpatchify_tokens
from cache_head_model_training import (
    CacheHeadTrainer,
    PromptDataset,
    dmd_loss,
    flow_to_x0,
    forward_diffuse,
    id_hash_split,
    prompt_split_checksum,
    regression_loss,
)
from fake_score_wan import FakeScoreWan


PATCH_SIZE = (1, 2, 2)
GRID = (2, 3, 4)          # f, h, w -> S = 24
IN_C = 16
TOK_C = 64
LATENT_SHAPE = (1, IN_C, GRID[0], GRID[1] * 2, GRID[2] * 2)


class FakeScheduler:
    def __init__(self, num_steps=15):
        self.num_train_timesteps = 1000.0
        self.set_timesteps(num_steps)

    def set_timesteps(self, num_steps, denoising_strength=1.0, shift=5.0):
        self.num_inference_steps = num_steps
        self.timesteps = torch.linspace(999.0, 1.0, num_steps)
        self.sigmas = self.timesteps / 1000.0

    def step(self, model_output, timestep, sample):
        return sample + 0.1 * model_output


class FakeDit(torch.nn.Module):
    """Fake Wan DiT: real Conv3d patchify -> tokens, context/time dependence,
    then unpatchify back to a latent velocity."""

    def __init__(self, grid=GRID, patch_size=PATCH_SIZE, in_c=IN_C, tok_c=TOK_C):
        super().__init__()
        self.grid = grid
        self.patch_size = patch_size
        self.in_c = in_c
        self.tok_c = tok_c
        self.patch_embedding = torch.nn.Conv3d(
            in_c, tok_c, kernel_size=patch_size, stride=patch_size, bias=False
        )
        # A genuine Linear so FakeScoreWan has LoRA targets to inject into.
        self.proj = torch.nn.Linear(tok_c, tok_c)
        self.calls = 0

    def forward(self, x, timestep, context, return_noise_tokens=False):
        self.calls += 1
        f, h, w = self.grid
        tokens = self.patch_embedding(x)
        tokens = einops.rearrange(tokens, "b d f h w -> b (f h w) d")
        t = timestep.float().reshape(-1, 1, 1) / 1000.0
        ctx = context.float().mean(dim=(1, 2), keepdim=True)
        tokens = tokens + t + ctx * 0.1 + 0.1 * self.proj(tokens)
        noise_pred = unpatchify_tokens(tokens, self.grid, self.patch_size)
        if return_noise_tokens:
            return noise_pred, tokens
        return noise_pred


class RecordingHead(CacheHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def forward(self, tokens, timestep, grid):
        self.calls += 1
        return super().forward(tokens, timestep, grid)


def _make_trainer(arm="residual_regression", lora_rank=2):
    torch.manual_seed(0)
    dit = FakeDit()
    scheduler = FakeScheduler(15)
    head = RecordingHead(CacheHeadConfig())
    fake = FakeScoreWan(dit, rank=lora_rank, alpha=1.0)
    dataset = [("0", "a dog runs"), ("1", "a cat sleeps"), ("2", "a bird flies")]
    schedule = CacheHeadSchedule(15, (1, 2, 6, 10, 14))
    trainer = CacheHeadTrainer(
        dit=dit, scheduler=scheduler, head=head, fake_score=fake,
        text_encode=lambda c: torch.randn(1, 4, 8),
        neg_ctx=torch.randn(1, 4, 8),
        dataset=dataset, schedule=schedule, cfg_scale=5.0,
        patch_size=PATCH_SIZE, grid=GRID, latent_shape=LATENT_SHAPE,
        arm=arm, device="cpu", dtype=torch.float32, batch_size=2,
        warmup_steps=2, updates=4, seed=0,
    )
    return trainer


# ═══════════════════════════════════════════════════════════════
# Flow / x0 helpers
# ═══════════════════════════════════════════════════════════════

def test_flow_to_x0_matches_formula():
    latents = torch.randn(2, 3, 4, 5)
    flow = torch.randn(2, 3, 4, 5)
    sigma = torch.tensor([0.3, 0.7])
    got = flow_to_x0(latents, flow, sigma)
    expected = latents - sigma.view(-1, 1, 1, 1) * flow
    assert torch.allclose(got, expected)


def test_forward_diffuse_matches_formula():
    x0 = torch.randn(2, 3, 4, 5)
    eps = torch.randn(2, 3, 4, 5)
    sigma = torch.tensor([0.3, 0.7])
    got = forward_diffuse(x0, eps, sigma)
    expected = (1 - sigma.view(-1, 1, 1, 1)) * x0 + sigma.view(-1, 1, 1, 1) * eps
    assert torch.allclose(got, expected)


def test_flow_to_x0_scalar_sigma():
    latents = torch.randn(1, 3, 4, 5)
    flow = torch.randn(1, 3, 4, 5)
    got = flow_to_x0(latents, flow, torch.tensor(0.5))
    assert torch.allclose(got, latents - 0.5 * flow)


# ═══════════════════════════════════════════════════════════════
# Loss formulas
# ═══════════════════════════════════════════════════════════════

def test_regression_loss_types():
    v = torch.randn(3, 4)
    t = torch.randn(3, 4)
    assert torch.allclose(regression_loss(v, t, "mse"), F.mse_loss(v, t))
    assert torch.allclose(regression_loss(v, t, "huber"), F.huber_loss(v, t, delta=1.0))
    with pytest.raises(ValueError):
        regression_loss(v, t, "nope")


def test_dmd_loss_matches_formula():
    x0 = torch.randn(2, 5)
    fake = torch.randn(2, 5)
    teacher = torch.randn(2, 5)
    w = 1.0 / ((x0 - teacher).abs().mean(dim=1, keepdim=True) + 1e-6)
    target = x0 - w * (fake - teacher)
    expected = 0.5 * F.mse_loss(x0, target)
    got = dmd_loss(x0, fake.detach(), teacher.detach())
    assert torch.allclose(got, expected)


def test_dmd_loss_zero_when_fake_equals_teacher():
    x0 = torch.randn(2, 5)
    loss = dmd_loss(x0, x0.detach(), x0.detach())
    assert loss.item() == 0.0


# ═══════════════════════════════════════════════════════════════
# Dataset splits
# ═══════════════════════════════════════════════════════════════

def test_id_hash_split_deterministic_and_counts(tmp_path):
    ids = [f"{i:04d}" for i in range(5000)]
    s1 = id_hash_split(ids)
    s2 = id_hash_split(ids)
    assert s1 == s2
    assert sum(len(s) for s in s1) == 5000
    assert all(len(set(s)) == len(s) for s in s1)  # no overlap within a split
    # distinct splits are disjoint
    assert set(s1[0]).isdisjoint(set(s1[1]))
    assert set(s1[0]).isdisjoint(set(s1[2]))
    assert set(s1[1]).isdisjoint(set(s1[2]))


def test_prompt_split_checksum_stable():
    ids = [f"{i:04d}" for i in range(100)]
    assert prompt_split_checksum(ids) == prompt_split_checksum(list(reversed(ids)))


# ═══════════════════════════════════════════════════════════════
# Gradient routing
# ═══════════════════════════════════════════════════════════════

def _param_groups(model):
    return {n: p for n, p in model.named_parameters()}


def test_regression_gradient_routing():
    trainer = _make_trainer("residual_regression")
    sample = trainer.sample_one()
    out = trainer.regression(sample)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    assert all(p.grad is not None for p in trainer.head.parameters())
    assert all(p.grad is None for p in trainer.dit.parameters())
    assert all(p.grad is None for p in trainer.fake_score.parameters())


def test_dmd_gradient_routing():
    trainer = _make_trainer("dmd")
    # Perturb the LoRA so fake != teacher and the DMD loss is non-zero.
    with torch.no_grad():
        for n, p in trainer.fake_score.named_parameters():
            if ".lora_B" in n:
                p.add_(0.05 * torch.randn_like(p))
    sample = trainer.sample_one()
    out = trainer.dmd(sample)
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() > 0
    out["loss"].backward()

    head_grads = [p.grad for p in trainer.head.parameters()]
    assert all(g is not None for g in head_grads)
    assert any(g.abs().sum() > 0 for g in head_grads)  # gradient reaches the head
    assert all(p.grad is None for p in trainer.dit.parameters())
    # Fake-score (base + lora) is queried under no_grad during the head update.
    assert all(p.grad is None for p in trainer.fake_score.parameters())


def test_fake_score_update_gradient_routing():
    trainer = _make_trainer("dmd")
    sample = trainer.sample_one()
    fs_loss = trainer.fake_score_update(sample)
    assert torch.isfinite(fs_loss)
    fs_loss.backward()

    lora_grads = [p.grad for p in trainer.fake_score.lora_parameters()]
    assert all(g is not None for g in lora_grads)
    assert any(g.abs().sum() > 0 for g in lora_grads)
    # Teacher and head are not part of the fake-score update.
    assert all(p.grad is None for p in trainer.dit.parameters())
    assert all(p.grad is None for p in trainer.head.parameters())
    # Frozen base of the clone gets no grads.
    assert all(p.grad is None for n, p in trainer.fake_score.base.named_parameters()
               if ".lora_A" not in n and ".lora_B" not in n)


def test_prefix_roll_call_counts():
    trainer = _make_trainer("residual_regression")
    latents = torch.randn(*LATENT_SHAPE)
    ctx = torch.randn(1, 4, 8)
    trainer.head.calls = 0
    trainer.dit.calls = 0
    trainer._prefix_roll(latents, ctx, stop_step=14)
    # Steps [0,14): full at 0,1,5,9,13 (5 x posi+nega = 10 dit calls); head at the rest (9).
    assert trainer.dit.calls == 10
    assert trainer.head.calls == 9


def test_train_few_steps_finite_and_checkpoint(tmp_path):
    trainer = _make_trainer("residual_regression")
    logs = trainer.train(save_dir=str(tmp_path), checkpoint_every=2, log_interval=1000)
    assert len(logs) == trainer.warmup_steps + trainer.updates
    assert all(rec["finite"] for rec in logs)
    assert (tmp_path / "cache_head_final.ckpt").is_file()
