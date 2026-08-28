"""CPU tests for the hybrid sampling loop (no Wan model needed)."""

import einops
import pytest
import torch

from cache_head_model import (
    CacheHead,
    CacheHeadConfig,
    CacheHeadSchedule,
    unpatchify_tokens,
)
from cache_head_model_inference import HybridSampler, full_step, head_step


PATCH_SIZE = (1, 2, 2)
GRID = (2, 3, 4)          # f, h, w -> S = 24
IN_C = 16
TOK_C = 64


class FakeScheduler:
    def __init__(self, num_steps=15):
        self.num_inference_steps = num_steps
        self.set_timesteps(num_steps)

    def set_timesteps(self, num_steps, denoising_strength=1.0, shift=5.0):
        # Decreasing timesteps like the Wan FlowMatchScheduler.
        self.num_inference_steps = num_steps
        self.timesteps = torch.linspace(999.0, 1.0, num_steps)
        self.sigmas = self.timesteps / 1000.0

    def step(self, model_output, timestep, sample):
        return sample + 0.1 * model_output


class FakeDit(torch.nn.Module):
    """Deterministic fake Wan DiT: real Conv3d patchify of the latent into
    [B, S, 64] noise tokens, context/time dependence, then unpatchify back to a
    latent velocity.  Latent shape is [B, C, F, 2h, 2w] for token grid (F, h, w)."""

    def __init__(self, grid=GRID, patch_size=PATCH_SIZE, in_c=IN_C, tok_c=TOK_C):
        super().__init__()
        self.grid = grid
        self.patch_size = patch_size
        self.in_c = in_c
        self.tok_c = tok_c
        self.patch_embedding = torch.nn.Conv3d(
            in_c, tok_c, kernel_size=patch_size, stride=patch_size, bias=False
        )
        self.calls = 0

    def forward(self, x, timestep, context, return_noise_tokens=False):
        self.calls += 1
        f, h, w = self.grid
        tokens = self.patch_embedding(x)  # [B, 64, F, h, w]
        tokens = einops.rearrange(tokens, "b d f h w -> b (f h w) d")
        # Depend on timestep and context so posi/nega differ under CFG.
        t = timestep.float().reshape(-1, 1, 1) / 1000.0
        ctx = context.float().mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        tokens = tokens + t + ctx * 0.1
        noise_pred = unpatchify_tokens(tokens, self.grid, self.patch_size)
        if return_noise_tokens:
            return noise_pred, tokens
        return noise_pred


class SpyHead(CacheHead):
    """CacheHead that records every (prev_guided, timestep) it sees."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def forward(self, tokens, timestep, grid):
        self.seen.append((tokens.detach().clone(), float(timestep.reshape(-1)[0])))
        return super().forward(tokens, timestep, grid)


def _make_dit_scheduler():
    return FakeDit(), FakeScheduler(15)


def _latent(seed=0):
    f, h, w = GRID
    torch.manual_seed(seed)
    return torch.randn(1, IN_C, f, h * 2, w * 2)


def _ctx(seed, B=1, L=5, D=8):
    torch.manual_seed(seed)
    return torch.randn(B, L, D)


def test_hybrid_schedule_call_counts():
    dit, scheduler = _make_dit_scheduler()
    head = SpyHead(CacheHeadConfig())
    schedule = CacheHeadSchedule(15, (1, 2, 6, 10, 14))
    sampler = HybridSampler(dit, scheduler, head, schedule, cfg_scale=5.0, patch_size=PATCH_SIZE, grid=GRID)
    latents = _latent()
    ctx_posi, ctx_nega = _ctx(1), _ctx(2)
    final, states, stats = sampler.sample(latents, ctx_posi, ctx_nega)

    assert stats["full_calls"] == 5
    assert stats["head_calls"] == 10
    assert dit.calls == 5 * 2  # 5 full steps x (posi + nega)
    assert len(states) == 16
    assert final.shape == latents.shape
    # Head steps are exactly the 10 non-anchor 0-indexed positions.
    head_steps = [i for i in range(15) if not schedule.is_full_step(i)]
    assert head_steps == [2, 3, 4, 6, 7, 8, 10, 11, 12, 14]


def test_prev_guided_propagation():
    """Each head step sees exactly the immediately preceding guided/v tokens.

    Recomputes guided_tokens from the recorded latent states (the dit is
    deterministic) and chains head-step v_tokens from the recorded inputs, then
    checks the spy's recorded head inputs against that expected chain."""
    dit, scheduler = _make_dit_scheduler()
    head = SpyHead(CacheHeadConfig())
    schedule = CacheHeadSchedule(15, (1, 2, 6, 10, 14))
    sampler = HybridSampler(dit, scheduler, head, schedule, cfg_scale=5.0, patch_size=PATCH_SIZE, grid=GRID)
    ctx_posi, ctx_nega = _ctx(1), _ctx(2)
    _, states, stats = sampler.sample(_latent(), ctx_posi, ctx_nega)

    seen = list(head.seen)  # snapshot; head() calls below would append
    assert len(seen) == stats["head_calls"] == 10

    expected_prev = None
    seen_idx = 0
    for progress_id in range(15):
        t = torch.tensor([scheduler.timesteps[progress_id].item()])
        if schedule.is_full_step(progress_id):
            # states[k] is the latent at the start of step k (float32 CPU).
            _, expected_prev = full_step(dit, states[progress_id], t, ctx_posi, ctx_nega, 5.0)
        else:
            seen_input, seen_t = seen[seen_idx]
            seen_idx += 1
            assert torch.equal(seen_input, expected_prev), f"head step {progress_id} saw wrong tokens"
            assert seen_t == pytest.approx(float(scheduler.timesteps[progress_id].item()))
            # Expected prev for the next head step = v_tokens = input + residual.
            residual = head(seen_input, torch.tensor([seen_t]), GRID)
            expected_prev = seen_input + residual
    assert seen_idx == 10


def test_zero_init_head_equals_carry_previous():
    """A zero-init head must reproduce carry_previous latents exactly."""
    dit, scheduler = _make_dit_scheduler()
    schedule = CacheHeadSchedule(15, (1, 2, 6, 10, 14))
    latents = _latent()
    ctx_posi, ctx_nega = _ctx(1), _ctx(2)

    # Reference: carry_previous keeps the nearest preceding guided tokens.
    def reference_carry(latents):
        x = latents
        prev_guided = None
        for progress_id in range(15):
            t = torch.tensor([scheduler.timesteps[progress_id].item()])
            if schedule.is_full_step(progress_id):
                noise_pred, prev_guided = full_step(dit, x, t, ctx_posi, ctx_nega, 5.0)
            else:
                noise_pred = unpatchify_tokens(prev_guided, GRID, PATCH_SIZE)
            x = scheduler.step(noise_pred, t, x)
        return x

    zero_head = CacheHead(CacheHeadConfig())
    sampler = HybridSampler(dit, scheduler, zero_head, schedule, cfg_scale=5.0, patch_size=PATCH_SIZE, grid=GRID)
    final_hybrid, _, _ = sampler.sample(latents, ctx_posi, ctx_nega)
    final_carry = reference_carry(latents)

    assert torch.equal(final_hybrid, final_carry)


def test_full_mode_all_full_calls():
    dit, scheduler = _make_dit_scheduler()
    head = SpyHead(CacheHeadConfig())
    schedule = CacheHeadSchedule(15, tuple(range(1, 16)))
    sampler = HybridSampler(dit, scheduler, head, schedule, cfg_scale=5.0, patch_size=PATCH_SIZE, grid=GRID)
    final, states, stats = sampler.sample(_latent(), _ctx(1), _ctx(2))
    assert stats["full_calls"] == 15
    assert stats["head_calls"] == 0
    assert dit.calls == 30
    assert len(states) == 16


# ═══════════════════════════════════════════════════════════════
# --checkpoint must not silently degrade to carry_previous
# ═══════════════════════════════════════════════════════════════

def test_missing_checkpoint_raises_instead_of_falling_back(tmp_path):
    """A wrong --checkpoint path used to load a zero-init head silently, which
    makes every checkpoint produce byte-identical output."""
    from cache_head_model_inference import check_checkpoint
    from cache_head_model import CacheHead, CacheHeadConfig, save_cache_head

    cfg = CacheHeadConfig()
    save_cache_head(CacheHead(cfg), cfg, tmp_path / "cache_head_step-200.ckpt")

    with pytest.raises(FileNotFoundError) as exc:
        check_checkpoint(str(tmp_path / "cache_head_200.ckpt"))
    # The message must name what is actually on disk.
    assert "cache_head_step-200.ckpt" in str(exc.value)


def test_existing_checkpoint_and_no_checkpoint_both_pass(tmp_path):
    from cache_head_model_inference import check_checkpoint
    from cache_head_model import CacheHead, CacheHeadConfig, save_cache_head

    cfg = CacheHeadConfig()
    path = tmp_path / "cache_head_final.ckpt"
    save_cache_head(CacheHead(cfg), cfg, path)
    check_checkpoint(str(path))   # present -> fine
    check_checkpoint(None)        # omitted -> zero-init baseline is intentional


def test_missing_checkpoint_in_missing_directory_still_raises(tmp_path):
    from cache_head_model_inference import check_checkpoint

    with pytest.raises(FileNotFoundError, match="No .ckpt files"):
        check_checkpoint(str(tmp_path / "nope" / "cache_head_final.ckpt"))
