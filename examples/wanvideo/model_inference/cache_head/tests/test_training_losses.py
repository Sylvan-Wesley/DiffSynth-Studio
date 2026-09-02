"""CPU tests for the CacheHead training losses and gradient routing.

Uses tiny fake dit/scheduler/head so no Wan model or GPU is needed.
"""

import einops
import pytest
import torch
import torch.nn.functional as F

from cache_head_model import (
    CacheHead, CacheHeadConfig, CacheHeadSchedule, patchify_latents, unpatchify_tokens,
)
from cache_head_model_inference import full_step
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

    def forward(self, tokens, timestep, grid, latent_tokens=None):
        self.calls += 1
        return super().forward(
            tokens, timestep, grid, latent_tokens=latent_tokens
        )


def _make_trainer(arm="residual_regression", lora_rank=2, **overrides):
    torch.manual_seed(0)
    dit = FakeDit()
    scheduler = FakeScheduler(15)
    head_variant = overrides.pop("head_variant", "legacy")
    head = RecordingHead(CacheHeadConfig(
        head_variant=head_variant,
        version=2 if head_variant == "legacy" else 3,
    ))
    # Non-DMD arms run without the fake-score clone entirely.
    fake = FakeScoreWan(dit, rank=lora_rank, alpha=1.0) if arm in ("dmd", "dmd_plus_reg") else None
    dataset = [("0", "a dog runs"), ("1", "a cat sleeps"), ("2", "a bird flies")]
    schedule = CacheHeadSchedule()
    def encode(caption):
        value = sum(ord(char) for char in caption) / 1000.0
        return torch.full((1, 4, 8), value)
    kwargs = dict(
        dit=dit, scheduler=scheduler, head=head, fake_score=fake,
        text_encode=encode,
        neg_ctx=torch.randn(1, 4, 8),
        dataset=dataset, schedule=schedule, cfg_scale=5.0,
        patch_size=PATCH_SIZE, grid=GRID, latent_shape=LATENT_SHAPE,
        arm=arm, device="cpu", dtype=torch.float32, batch_size=2,
        warmup_steps=2, updates=4, seed=0,
    )
    kwargs.update(overrides)
    return CacheHeadTrainer(**kwargs)


def _perturb_head(trainer, scale=0.05):
    """Move a zero-init head away from the carry-previous baseline."""
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
    torch.manual_seed(0)
    latents = torch.randn(2, 3, 4, 5)
    flow = torch.randn(2, 3, 4, 5)
    sigma = torch.tensor([0.3, 0.7])
    got = flow_to_x0(latents, flow, sigma)
    expected = latents - sigma.view(-1, 1, 1, 1) * flow
    # The helper accumulates in float64 and casts back, so it disagrees with a
    # pure-float32 formula by ~1e-7 -- above allclose's default atol of 1e-8
    # wherever an element lands near zero.
    assert torch.allclose(got, expected, atol=1e-6)


def test_forward_diffuse_matches_formula():
    torch.manual_seed(0)
    x0 = torch.randn(2, 3, 4, 5)
    eps = torch.randn(2, 3, 4, 5)
    sigma = torch.tensor([0.3, 0.7])
    got = forward_diffuse(x0, eps, sigma)
    expected = (1 - sigma.view(-1, 1, 1, 1)) * x0 + sigma.view(-1, 1, 1, 1) * eps
    # See test_flow_to_x0_matches_formula: float64 accumulation vs a float32
    # reference needs a tolerance above the default atol.
    assert torch.allclose(got, expected, atol=1e-6)


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
    # Steps [0,14): full at 0--6 (7 x posi+nega); head at 7--13.
    assert trainer.dit.calls == 14
    assert trainer.head.calls == 7


def test_train_few_steps_finite_and_checkpoint(tmp_path):
    trainer = _make_trainer("residual_regression")
    logs = trainer.train(save_dir=str(tmp_path), checkpoint_every=2, log_interval=1000)
    assert len(logs) == trainer.warmup_steps + trainer.updates
    assert all(rec["finite"] for rec in logs)
    # The student's pre-clip gradient norm rides along on every step record; the
    # fake score never runs on this arm, so its norm stays NaN.
    assert all(torch.isfinite(torch.tensor(rec["grad_norm"])) for rec in logs)
    assert all(torch.isnan(torch.tensor(rec["fake_grad_norm"])) for rec in logs)
    assert (tmp_path / "cache_head_final.ckpt").is_file()


def test_train_mirrors_progress_lines_into_run_log(tmp_path):
    trainer = _make_trainer("residual_regression")
    log_path = tmp_path / "logs" / "residual_regression-20260828-120000+0000.txt"
    log_path.parent.mkdir()
    trainer.train(save_dir=None, checkpoint_every=0, log_interval=1, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == trainer.warmup_steps + trainer.updates
    assert all("grad_norm=" in line for line in lines)


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


def test_optimizer_steps_per_iteration_must_be_positive():
    with pytest.raises(ValueError, match="optimizer_steps_per_iteration"):
        _make_trainer("supervised", optimizer_steps_per_iteration=0)


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


def test_supervised_trajectory_uses_teacher_at_all_denoising_steps():
    trainer = _make_trainer("supervised")
    trainer.dit.calls = 0
    trainer.head.calls = 0
    out = trainer.supervised_trajectory(_batch(trainer, 1))
    # Full teacher CFG runs at all 15 steps; the head trains only on 8--15.
    assert trainer.dit.calls == 30
    assert trainer.head.calls == trainer.schedule.num_head_steps == 8
    assert [rec["step"] for rec in out["per_step"]] == [
        i for i in trainer.schedule.head_step_indices
    ]


def test_supervised_trajectory_teacher_forces_every_student_input():
    trainer = _make_trainer("supervised")
    _perturb_head(trainer)
    batch = _batch(trainer, 1)
    teacher = trainer.teacher_guided_trajectory(batch)
    seen = []
    original = trainer.head.forward

    def spy(tokens, timestep, grid):
        seen.append(tokens.detach().clone())
        return original(tokens, timestep, grid)

    trainer.head.forward = spy
    trainer.supervised_teacher_forced(teacher)
    expected = [teacher[:, step - 2] for step in trainer.schedule.head_step_indices]
    assert len(seen) == len(expected)
    assert all(torch.equal(actual, target) for actual, target in zip(seen, expected))
    assert all(not actual.requires_grad for actual in seen)


def test_supervised_loss_is_computed_on_tokens(monkeypatch):
    trainer = _make_trainer("supervised")
    wrong = torch.randn(1, 15, *LATENT_SHAPE[1:])
    with pytest.raises(RuntimeError, match="teacher-guided trajectory"):
        trainer.supervised_teacher_forced(wrong)


def test_supervised_loss_matches_manual_token_space_loss():
    trainer = _make_trainer("supervised")
    _perturb_head(trainer)
    teacher = trainer.teacher_guided_trajectory(_batch(trainer, 1))
    out = trainer.supervised_teacher_forced(teacher)
    losses = []
    for step in trainer.schedule.head_step_indices:
        previous = teacher[:, step - 2]
        target = teacher[:, step - 1]
        student = previous + trainer.head(previous, trainer._t(step - 1), trainer.grid)
        losses.append(F.mse_loss(student.float(), target.float()))
    manual = torch.stack(losses).mean()
    assert torch.allclose(out["loss"], manual)


def test_reconstructed_teacher_latents_match_direct_rollout():
    trainer = _make_trainer("supervised", head_variant="latent_residual")
    batch = _batch(trainer, 2)
    teacher = trainer.teacher_guided_trajectory(batch)
    reconstructed = trainer.reconstruct_teacher_latents(batch["z"], teacher)

    latents = batch["z"]
    direct_starts = []
    for k in range(trainer.schedule.num_inference_steps):
        direct_starts.append(latents)
        noise, _ = full_step(
            trainer.dit, latents, trainer._t(k), batch["ctx"],
            trainer._neg_ctx_for(latents.shape[0]), trainer.cfg,
        )
        latents = trainer.scheduler.step(noise, trainer._t(k), latents)
    direct = torch.stack(direct_starts, dim=1)
    assert torch.equal(reconstructed, direct)


@pytest.mark.parametrize(
    "variant", ["latent_fusion", "latent_residual", "latent_residual_deep"]
)
def test_supervised_latent_variants_receive_xk_previous_velocity_and_vk(variant):
    trainer = _make_trainer("supervised", head_variant=variant)
    _perturb_head(trainer)
    batch = _batch(trainer, 1)
    teacher = trainer.teacher_guided_trajectory(batch)
    reconstructed = trainer.reconstruct_teacher_latents(batch["z"], teacher)
    seen = []
    original = trainer.head.forward

    def spy(tokens, timestep, grid, latent_tokens=None):
        seen.append((tokens.detach().clone(), latent_tokens.detach().clone()))
        return original(tokens, timestep, grid, latent_tokens=latent_tokens)

    trainer.head.forward = spy
    out = trainer.supervised_teacher_forced(teacher, batch["z"])
    for (seen_previous, seen_latent), rec in zip(seen, out["per_step"]):
        k = rec["step"] - 1
        previous = teacher[:, k - 1]
        target = teacher[:, k]
        assert torch.equal(seen_previous, previous)
        assert torch.equal(
            seen_latent, patchify_latents(reconstructed[:, k], GRID, PATCH_SIZE)
        )
        carry = F.mse_loss(previous.float(), target.float()).item()
        assert rec["carry_mse"] == pytest.approx(carry)
        assert rec["relative_improvement"] == pytest.approx(
            1.0 - rec["loss"] / carry, rel=1e-5
        )


def test_latent_supervision_requires_initial_latents():
    trainer = _make_trainer("supervised", head_variant="latent_fusion")
    teacher = trainer.teacher_guided_trajectory(_batch(trainer, 1))
    with pytest.raises(ValueError, match="requires deterministic initial_latents"):
        trainer.supervised_teacher_forced(teacher)


def test_supervised_batch_matches_two_single_rollouts():
    trainer = _make_trainer("supervised")
    _perturb_head(trainer)
    batch = _batch(trainer, 2)
    teacher = trainer.teacher_guided_trajectory(batch)
    both = trainer.supervised_teacher_forced(teacher)

    singles = []
    for i in range(2):
        singles.append(trainer.supervised_teacher_forced(teacher[i:i + 1])["loss"].item())
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
        val_dataset=val, val_batches=1, optimizer_steps_per_iteration=2,
        trajectory_dir=tmp_path / "trajectories",
    )
    _perturb_head(trainer)
    logs = trainer.train(save_dir=str(tmp_path), checkpoint_every=0, log_interval=1000)

    train_records = [r for r in logs if r["phase"] == "supervised"]
    val_records = [r for r in logs if r["phase"] == "val"]
    assert train_records and all(r["finite"] for r in train_records)
    assert all(torch.isfinite(torch.tensor(r["grad_norm"])) for r in train_records)
    # One data iteration per epoch, with two optimizer updates per iteration.
    assert len(train_records) == 4
    assert len(val_records) == 2
    assert all(torch.isfinite(torch.tensor(r["val_loss"])) for r in val_records)
    # Per-head-step detail is what the heat map is read against.
    assert set(val_records[0]["val_per_step"]) == {
        i for i in trainer.schedule.head_step_indices
    }
    assert (tmp_path / "cache_head_final.ckpt").is_file()
    assert (tmp_path / "cache_head_best.ckpt").is_file()


def test_teacher_trajectory_cache_is_lazy_persistent_and_guided(tmp_path):
    trainer = _make_trainer("supervised", trajectory_dir=tmp_path)
    item = [trainer.dataset[0]]
    trainer.dit.calls = 0
    first = trainer.load_teacher_batch(item, split="train")
    assert first.shape == (1, 15, GRID[0] * GRID[1] * GRID[2], TOK_C)
    assert trainer.dit.calls == 30
    cache_files = list(tmp_path.rglob("*.pt"))
    assert len(cache_files) == 1
    payload = torch.load(cache_files[0], weights_only=False)
    assert set(payload) == {"metadata", "guided_tokens"}
    assert torch.equal(payload["guided_tokens"], first[0])

    trainer.dit.calls = 0
    second = trainer.load_teacher_batch(item, split="train")
    assert trainer.dit.calls == 0
    assert torch.equal(second, first)
    assert trainer.trajectory_cache_hits == 1
    assert trainer.trajectory_cache_misses == 1


def test_cached_trajectory_uses_same_regenerated_caption_seed(tmp_path):
    trainer = _make_trainer(
        "supervised", trajectory_dir=tmp_path, head_variant="latent_residual"
    )
    items = [trainer.dataset[0], trainer.dataset[1]]
    cached = trainer.load_teacher_batch(items, split="train")
    initial = trainer.deterministic_initial_latents(items, split="train")
    direct = trainer.teacher_guided_trajectory({
        "z": initial,
        "ctx": trainer.encode_captions([caption for _, caption in items]),
    })
    assert torch.equal(cached, direct)


def test_default_five_optimizer_updates_are_logged_and_sent_to_wandb(tmp_path):
    class Run:
        def __init__(self):
            self.records = []

        def log(self, record, step=None):
            self.records.append((step, record))

    trainer = _make_trainer(
        "supervised", batch_size=2, micro_batch=1, epochs=1,
        trajectory_dir=tmp_path / "trajectories",
    )
    run = Run()
    logs = trainer.train(
        save_dir=None, checkpoint_every=0, log_interval=1000,
        wandb_run=run, log_path=tmp_path / "training.txt",
    )
    train_records = [record for record in logs if record["phase"] == "supervised"]
    assert len(train_records) == 5
    assert [record["inner_step"] for record in train_records] == list(range(5))
    assert [step for step, _ in run.records] == list(range(5))
    assert len((tmp_path / "training.txt").read_text().splitlines()) == 5


def test_supervised_epoch_needs_enough_prompts(tmp_path):
    trainer = _make_trainer(
        "supervised", batch_size=8, micro_batch=2, trajectory_dir=tmp_path
    )
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


def test_no_network_disables_wandb_even_when_configured(monkeypatch):
    _clear_offline_env(monkeypatch)
    args = argparse.Namespace(
        no_network=True, model_base_path=None,
        wandb_project="cache-head", wandb_mode="offline",
    )
    apply_no_network(args)
    assert args.wandb_project is None


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
