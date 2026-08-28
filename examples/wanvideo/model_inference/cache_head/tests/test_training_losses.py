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
    apply_no_network,
    training_type_for_arm,
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

    def forward(self, x, timestep, context, return_noise_tokens=False, **kwargs):
        # The real WanModel.forward accepts **kwargs (use_gradient_checkpointing,
        # RAS knobs, ...); the double has to as well or fake_score_update fails.
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


def _make_trainer(arm="residual_regression", lora_rank=2, **overrides):
    torch.manual_seed(0)
    dit = FakeDit()
    scheduler = FakeScheduler(15)
    head = RecordingHead(CacheHeadConfig())
    # Non-DMD arms run without the fake-score clone entirely.
    fake = FakeScoreWan(dit, rank=lora_rank, alpha=1.0) if arm in ("dmd", "dmd_plus_reg") else None
    dataset = [("0", "a dog runs"), ("1", "a cat sleeps"), ("2", "a bird flies")]
    schedule = CacheHeadSchedule(15, (1, 2, 6, 10, 14))
    kwargs = dict(
        dit=dit, scheduler=scheduler, head=head, fake_score=fake,
        text_encode=lambda c: torch.randn(1, 4, 8),
        neg_ctx=torch.randn(1, 4, 8),
        dataset=dataset, schedule=schedule, cfg_scale=5.0,
        patch_size=PATCH_SIZE, grid=GRID, latent_shape=LATENT_SHAPE,
        arm=arm, device="cpu", dtype=torch.float32, batch_size=2,
        warmup_steps=2, updates=4, seed=0,
    )
    kwargs.update(overrides)
    return CacheHeadTrainer(**kwargs)


def _perturb_head(trainer, scale=0.05):
    """The head is zero-init (residual == 0); give it a non-trivial output."""
    with torch.no_grad():
        trainer.head.out_proj.weight.add_(scale * torch.randn_like(trainer.head.out_proj.weight))


def _batch(trainer, n, seed=0):
    """A rollout batch with fixed latents/context (no RNG coupling to the trainer)."""
    torch.manual_seed(seed)
    return {
        "z": torch.randn(n, *LATENT_SHAPE[1:]),
        "ctx": torch.randn(n, 4, 8),
    }


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


# ═══════════════════════════════════════════════════════════════
# Supervised arm
# ═══════════════════════════════════════════════════════════════

def test_training_type_derived_from_arm():
    assert training_type_for_arm("supervised") == "supervised"
    for arm in ("residual_regression", "dmd", "dmd_plus_reg"):
        assert training_type_for_arm(arm) == "dmd"


def test_supervised_arm_needs_no_fake_score():
    trainer = _make_trainer("supervised")
    assert trainer.fake_score is None
    assert trainer.fake_opt is None


def test_dmd_arm_requires_fake_score():
    with pytest.raises(ValueError, match="fake-score"):
        _make_trainer("dmd", fake_score=None)


def test_batch_size_must_be_multiple_of_micro_batch():
    with pytest.raises(ValueError, match="multiple of micro_batch"):
        _make_trainer("supervised", batch_size=3, micro_batch=2)


def test_supervised_trajectory_gradient_routing():
    trainer = _make_trainer("supervised")
    _perturb_head(trainer)
    out = trainer.supervised_trajectory(_batch(trainer, 2))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    head_grads = [p.grad for p in trainer.head.parameters()]
    assert all(g is not None for g in head_grads)
    assert any(g.abs().sum() > 0 for g in head_grads)
    assert all(p.grad is None for p in trainer.dit.parameters())


def test_supervised_trajectory_supervises_every_head_step():
    trainer = _make_trainer("supervised")
    trainer.dit.calls = 0
    trainer.head.calls = 0
    out = trainer.supervised_trajectory(_batch(trainer, 1))
    # A teacher query at all 15 steps (posi+nega each) and the head at the 10 head steps.
    assert trainer.dit.calls == 30
    assert trainer.head.calls == trainer.schedule.num_head_steps == 10
    assert [rec["step"] for rec in out["per_step"]] == [
        i - 1 for i in trainer.schedule.head_step_indices
    ]


def test_supervised_trajectory_detaches_between_head_steps():
    """Without chaining, no head step's graph may reach a previous one."""
    trainer = _make_trainer("supervised", chain_run_grads=False)
    _perturb_head(trainer)
    seen = []
    original = trainer.head.forward

    def spy(tokens, timestep, grid):
        seen.append(tokens.requires_grad)
        return original(tokens, timestep, grid)

    trainer.head.forward = spy
    trainer.supervised_trajectory(_batch(trainer, 1))
    # Every head step receives a detached carry, including steps inside a run.
    assert seen and not any(seen)


def test_supervised_trajectory_chains_run_grads_when_enabled():
    trainer = _make_trainer("supervised", chain_run_grads=True)
    _perturb_head(trainer)
    seen = []
    original = trainer.head.forward

    def spy(tokens, timestep, grid):
        seen.append(tokens.requires_grad)
        return original(tokens, timestep, grid)

    trainer.head.forward = spy
    trainer.supervised_trajectory(_batch(trainer, 1))
    # Head steps 2 and 3 of each run consume a graph-carrying carry.
    assert any(seen)


def test_supervised_loss_is_computed_on_tokens(monkeypatch):
    """The [B, N, C] guard must reject the latent-shaped full_step return."""
    import cache_head_model_training as training

    trainer = _make_trainer("supervised")
    trainer.head.calls = 0
    real_full_step = training.full_step

    def swapped(*args, **kwargs):
        noise_pred, tokens = real_full_step(*args, **kwargs)
        # Corrupt only the teacher query at head steps -- that is the call whose
        # return the draft mis-bound.  Swapping the anchor steps too would break
        # the rollout before the guard ever runs.
        if trainer.head.calls >= 1:
            return tokens, noise_pred
        return noise_pred, tokens

    monkeypatch.setattr(training, "full_step", swapped)
    with pytest.raises(RuntimeError, match=r"\[B, N, C\]"):
        trainer.supervised_trajectory(_batch(trainer, 1))


def test_supervised_loss_matches_manual_token_space_loss():
    trainer = _make_trainer("supervised")
    _perturb_head(trainer)
    out = trainer.supervised_trajectory(_batch(trainer, 1))
    manual = sum(rec["loss"] for rec in out["per_step"]) / trainer.schedule.num_head_steps
    assert out["loss"].item() == pytest.approx(manual, rel=1e-5)


def test_supervised_batch_matches_two_single_rollouts():
    trainer = _make_trainer("supervised")
    _perturb_head(trainer)
    batch = _batch(trainer, 2)
    both = trainer.supervised_trajectory(batch)

    singles = []
    for i in range(2):
        one = {"z": batch["z"][i:i + 1], "ctx": batch["ctx"][i:i + 1]}
        singles.append(trainer.supervised_trajectory(one)["loss"].item())
    # MSE/Huber average over the batch, so the batched loss is the mean.
    assert both["loss"].item() == pytest.approx(sum(singles) / 2, rel=1e-4)


def test_train_rejects_unknown_training_type():
    trainer = _make_trainer("supervised")
    with pytest.raises(ValueError, match="training_type"):
        trainer.train(training_type="bogus", log_interval=1000)


def test_train_supervised_runs_epochs_and_validates(tmp_path):
    val = [("9", "a fox waits"), ("8", "a wave breaks")]
    trainer = _make_trainer(
        "supervised", batch_size=2, micro_batch=1, epochs=2,
        val_dataset=val, val_batches=1,
    )
    _perturb_head(trainer)
    logs = trainer.train(save_dir=str(tmp_path), checkpoint_every=0, log_interval=1000)

    train_records = [r for r in logs if r["phase"] == "supervised"]
    val_records = [r for r in logs if r["phase"] == "val"]
    assert train_records and all(r["finite"] for r in train_records)
    # 3 prompts, micro_batch 1 x accum 2 -> 1 optimizer step per epoch, 2 epochs.
    assert len(train_records) == 2
    assert len(val_records) == 2
    assert all(torch.isfinite(torch.tensor(r["val_loss"])) for r in val_records)
    # Per-head-step detail is what the heat map is read against.
    assert set(val_records[0]["val_per_step"]) == {
        i - 1 for i in trainer.schedule.head_step_indices
    }
    assert (tmp_path / "cache_head_final.ckpt").is_file()
    assert (tmp_path / "cache_head_best.ckpt").is_file()


def test_supervised_epoch_needs_enough_prompts():
    trainer = _make_trainer("supervised", batch_size=8, micro_batch=2)
    with pytest.raises(ValueError, match="at least"):
        trainer.train(log_interval=1000)


# ═══════════════════════════════════════════════════════════════
# Offline (--no-network) mode
# ═══════════════════════════════════════════════════════════════

import argparse
import os


def _clear_offline_env(monkeypatch):
    for key in ("DIFFSYNTH_SKIP_DOWNLOAD", "DIFFSYNTH_MODEL_BASE_PATH",
                "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)


def test_no_network_is_a_noop_when_disabled(monkeypatch):
    _clear_offline_env(monkeypatch)
    apply_no_network(argparse.Namespace(no_network=False, model_base_path=None))
    assert "DIFFSYNTH_SKIP_DOWNLOAD" not in os.environ
    assert "HF_HUB_OFFLINE" not in os.environ


def test_no_network_disables_downloads_and_hub_lookups(monkeypatch):
    _clear_offline_env(monkeypatch)
    apply_no_network(argparse.Namespace(no_network=True, model_base_path="/data/wan"))
    # ModelConfig.parse_skip_download() reads this and short-circuits download().
    assert os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] == "true"
    # ModelConfig.reset_local_model_path() reads this to resolve <base>/<model_id>/.
    assert os.environ["DIFFSYNTH_MODEL_BASE_PATH"] == "/data/wan"
    # The tokenizer loads via AutoTokenizer and would otherwise revision-check.
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_no_network_respects_an_existing_offline_setup(monkeypatch):
    """Never override what the operator already configured."""
    _clear_offline_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    apply_no_network(argparse.Namespace(no_network=True, model_base_path=None))
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    # ...but the download switch is ours to own.
    assert os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] == "true"


def test_no_network_leaves_model_base_path_unset_when_not_given(monkeypatch):
    _clear_offline_env(monkeypatch)
    apply_no_network(argparse.Namespace(no_network=True, model_base_path=None))
    # ModelConfig falls back to its own default (./models).
    assert "DIFFSYNTH_MODEL_BASE_PATH" not in os.environ
