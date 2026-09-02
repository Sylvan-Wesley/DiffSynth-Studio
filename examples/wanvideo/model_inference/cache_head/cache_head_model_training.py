"""
CacheHead training harness + loss study.

The supervised arm samples a fixed full-Wan teacher trajectory, persists all
15 guided CFG token predictions, and trains only on the schedule complement.
At student step ``k``, teacher tokens from ``k-1`` are the velocity input and
teacher tokens from ``k`` are the MSE target.  Version-3 heads also receive the
current teacher latent, reconstructed from the deterministic initial latent
and cached velocities.  The teacher velocity advances every denoising step,
so student predictions never alter the training trajectory.
Each data iteration repeats this all-student-step update a configurable number
of times (five by default).

Arms:
    carry_previous       no learned head
    residual_regression  Huber/MSE to frozen-Wan velocity at the hybrid state
    dmd                  Strict DMD after a shared regression warm-up
    dmd_plus_reg         DMD + regression (sweep ``--reg-weight`` 0.03/0.1/0.3)

The legacy DMD arms use a training-only LoRA-Wan fake-score estimator
(``fake_score_wan.py``) on top of frozen Wan's score prediction; one CacheHead
update alternates with five fake-score updates.  The fake-score is discarded
after training; only CacheHead weights + config are exported.

Loss conventions follow ``diffsynth/diffusion/dmd2.py``:
    flow_to_x0(latents, flow, sigma) = latents - sigma*flow
    weight w = 1/(|x0 - teacher_x0|.abs().mean + 1e-6)
    L_DMD = 0.5*|x0 - sg[x0 - w*(fake_x0 - teacher_x0)]|^2
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from cache_head_model import (
    WAN_NOISE_TOKEN_CHANNELS,
    CacheHead,
    CacheHeadConfig,
    CacheHeadSchedule,
    HEAD_VARIANTS,
    parse_full_step_indices,
    patchify_latents,
    save_cache_head,
    unpatchify_tokens,
)
from cache_head_model_inference import full_step, head_step
from cache_head_ddp import DistributedContext, initialize_distributed
from fake_score_wan import FakeScoreWan


# Arms that train a head.  ``carry_previous`` is the explicit zero-residual
# baseline and is never trained.  The loop an arm runs is derived from the arm itself rather
# than a second, orthogonal switch.
DMD_ARMS = ("dmd", "dmd_plus_reg")
TRAINING_ARMS = ("residual_regression", "supervised") + DMD_ARMS
TRAINING_TYPES = ("dmd", "supervised")


def training_type_for_arm(arm: str) -> str:
    """The training loop an arm runs: the epoch/validation loop for
    ``supervised``, the step-based DMD loop for everything else."""
    return "supervised" if arm == "supervised" else "dmd"


# ═══════════════════════════════════════════════════════════════
# DMD2 flow/x0 conventions (mirrors diffsynth/diffusion/dmd2.py)
# ═══════════════════════════════════════════════════════════════

def _expand_like(value: torch.Tensor, target: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Broadcast ``value`` (scalar or [B]) to ``target``'s ndim, float64."""
    value = value.to(device=target.device, dtype=torch.float64)
    while value.ndim < target.ndim:
        value = value.unsqueeze(-1)
    return value


def flow_to_x0(latents: torch.Tensor, flow: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """x0 = latents - sigma*flow (float64 internally, repo convention)."""
    original_dtype = latents.dtype
    latents = latents.to(torch.float64)
    flow = flow.to(torch.float64)
    sigma = _expand_like(sigma, latents, torch.float64)
    return (latents - sigma * flow).to(original_dtype)


def forward_diffuse(x0: torch.Tensor, eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Flow forward process: x_t = (1-sigma)*x0 + sigma*eps."""
    original_dtype = x0.dtype
    x0 = x0.to(torch.float64)
    eps = eps.to(torch.float64)
    sigma = _expand_like(sigma, x0, torch.float64)
    return ((1 - sigma) * x0 + sigma * eps).to(original_dtype)


def regression_loss(v_hat: torch.Tensor, teacher: torch.Tensor, loss_type: str = "huber") -> torch.Tensor:
    if loss_type == "huber":
        return F.huber_loss(v_hat, teacher, delta=1.0)
    if loss_type == "mse":
        return F.mse_loss(v_hat, teacher)
    raise ValueError(f"unknown regression loss {loss_type!r} (use huber|mse)")


def dmd_loss(x0: torch.Tensor, fake_x0: torch.Tensor, teacher_x0: torch.Tensor) -> torch.Tensor:
    """Repo DMD2 gradient-normalization convention.

    ``fake_x0`` and ``teacher_x0`` must already be stop-gradients (computed
    under no_grad by the caller).  ``w`` is the per-sample inverse mean-abs
    distance used by ``_variational_score_distillation_loss`` in dmd2.py.
    """
    dims = tuple(range(1, teacher_x0.ndim))
    with torch.no_grad():
        weight = 1 / ((x0 - teacher_x0).abs().mean(dim=dims, keepdim=True) + 1e-6)
        weight = weight.to(dtype=x0.dtype)
        pseudo_target = x0 - weight * (fake_x0 - teacher_x0)
    pseudo_target = pseudo_target.detach()
    return 0.5 * F.mse_loss(x0, pseudo_target)


# ═══════════════════════════════════════════════════════════════
# MixKit caption dataset with deterministic ID-hash splits
# ═══════════════════════════════════════════════════════════════

def id_hash_split(caption_ids: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Deterministic 84.6% / 7.7% / 7.7% train / val / test hash split."""
    train, val, test = [], [], []
    for cid in caption_ids:
        h = int(hashlib.sha256(str(cid).encode("utf-8")).hexdigest(), 16) % 1000
        if h < 846:
            train.append(cid)
        elif h < 923:
            val.append(cid)
        else:
            test.append(cid)
    return train, val, test


def prompt_split_checksum(caption_ids: list[str]) -> str:
    """sha256 over the deterministic (id, split) assignment."""
    splits = id_hash_split(caption_ids)
    split_of = {}
    for split, ids in zip(("train", "val", "test"), splits):
        for cid in ids:
            split_of[cid] = split
    payload = "".join(f"{cid}:{split_of[cid]};" for cid in sorted(caption_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PromptDataset(torch.utils.data.Dataset):
    """JSONL prompt split, deterministically sharded by DDP rank."""

    def __init__(
        self,
        captions_path: str,
        split: str = "train",
        subset: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        items: list[tuple[str, str]] = []
        with open(captions_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                items.append((str(rec["id"]), rec["caption"]))
        ids = sorted(cid for cid, _ in items)
        split_ids = {"train": id_hash_split(ids)[0],
                     "val": id_hash_split(ids)[1],
                     "test": id_hash_split(ids)[2]}[split]
        by_id = dict(items)
        self.items = [(cid, by_id[cid]) for cid in split_ids]
        if subset is not None:
            self.items = self.items[:subset]
        self.items = self.items[rank::world_size]
        if not self.items:
            raise ValueError(
                f"DDP rank {rank} received no prompts. Increase --subset or reduce WORLD_SIZE."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, str]:
        return self.items[index]


# ═══════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════

class CacheHeadTrainer:
    def __init__(
        self,
        *,
        dit: nn.Module,
        scheduler,
        head: CacheHead,
        fake_score: FakeScoreWan | None,
        text_encode: Callable[[str], torch.Tensor],
        neg_ctx: torch.Tensor,
        dataset: PromptDataset,
        schedule: CacheHeadSchedule,
        cfg_scale: float,
        patch_size,
        grid: tuple[int, int, int],
        latent_shape: tuple[int, int, int, int, int],
        arm: str,
        reg_loss: str = "huber",
        reg_weight: float = 0.1,
        lr: float = 1e-4,
        lora_lr: float = 1e-4,
        batch_size: int = 8,
        micro_batch: int = 1,
        grad_clip: float = 1.0,
        warmup_steps: int = 2000,
        updates: int = 10000,
        epochs: int = 1,
        optimizer_steps_per_iteration: int = 5,
        trajectory_dir: str | Path | None = None,
        trajectory_seed: int | None = None,
        text_encode_batch: Callable[[list[str]], torch.Tensor] | None = None,
        val_dataset: PromptDataset | None = None,
        val_batches: int = 8,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        sigma_min: float = 0.001,
        sigma_max: float = 0.999,
        seed: int = 0,
        distributed: DistributedContext | None = None,
    ):
        if arm not in TRAINING_ARMS:
            raise ValueError(
                f"arm must be one of {TRAINING_ARMS}, got {arm!r} (carry_previous is not trained)"
            )
        if batch_size % micro_batch != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a multiple of micro_batch ({micro_batch}) "
                f"so the effective batch is exactly what was requested"
            )
        if arm in DMD_ARMS and fake_score is None:
            raise ValueError(f"arm {arm!r} needs a fake-score estimator, got fake_score=None")
        if optimizer_steps_per_iteration < 1:
            raise ValueError(
                "optimizer_steps_per_iteration must be positive, got "
                f"{optimizer_steps_per_iteration}"
            )
        self.dit = dit
        self.scheduler = scheduler
        self.head = head
        self.fake_score = fake_score
        self.text_encode = text_encode
        self.neg_ctx = neg_ctx
        self.dataset = dataset
        self.schedule = schedule
        self.cfg = cfg_scale
        self.patch = patch_size
        self.grid = grid
        self.latent_shape = latent_shape
        self.arm = arm
        self.reg_loss = reg_loss
        self.reg_weight = reg_weight
        self.batch_size = batch_size
        self.micro_batch = micro_batch
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.updates = updates
        self.epochs = epochs
        self.optimizer_steps_per_iteration = optimizer_steps_per_iteration
        self.trajectory_dir = Path(trajectory_dir) if trajectory_dir else None
        self.trajectory_seed = seed if trajectory_seed is None else trajectory_seed
        self.val_dataset = val_dataset
        self.val_batches = val_batches
        self.device = device
        self.dtype = dtype
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.seed = seed
        self.distributed = distributed or DistributedContext(
            rank=0, local_rank=0, world_size=1, device=torch.device(device)
        )

        # Batched text encoding falls back to per-caption encoding so callers
        # that only supply ``text_encode`` (the CPU tests) keep working.
        self._text_encode_batch = text_encode_batch

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        # Freeze the teacher; only CacheHead trains in the production optimizer.
        self.dit.eval()
        self.dit.requires_grad_(False)
        assert all(not p.requires_grad for p in self.dit.parameters()), "teacher Wan must be frozen"
        self.head.train()
        self.head_opt = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-2)
        # The fake-score estimator deep-copies the whole DiT, so the non-DMD
        # arms skip it entirely and spend that memory on a larger micro-batch.
        if self.fake_score is None:
            self.fake_score_module = None
            self.fake_opt = None
        else:
            self.fake_score_module = self.distributed.unwrap(self.fake_score)
            self.fake_opt = torch.optim.AdamW(
                list(self.fake_score_module.lora_parameters()), lr=lora_lr, weight_decay=1e-2
            )
            self.fake_score.eval()

        self.logs: list[dict] = []
        # Plain-text mirror of the progress lines; ``train`` fills it in.
        self.log_path: Path | None = None
        self._trajectory_fingerprint_value: str | None = None
        self.trajectory_cache_hits = 0
        self.trajectory_cache_misses = 0

    # ---- sampling --------------------------------------------------------

    def sample_one(self) -> dict:
        _, caption = self.dataset[self.rng.randrange(len(self.dataset))]
        ctx = self.text_encode(caption)
        z = torch.randn(*self.latent_shape, device=self.device, dtype=self.dtype)
        # 1-indexed head index -> 0-indexed progress id.
        step_j = self.rng.choice(self.schedule.head_step_indices) - 1
        return {"z": z, "ctx": ctx, "step_j": step_j}

    def encode_captions(self, captions: list[str]) -> torch.Tensor:
        """Context embeddings ``[B, L, D]`` for a list of captions."""
        if self._text_encode_batch is not None:
            return self._text_encode_batch(captions)
        return torch.cat([self.text_encode(c) for c in captions], dim=0)

    def make_batch(self, captions: list[str]) -> dict:
        """Build a batched rollout input from explicit captions."""
        ctx = self.encode_captions(captions)
        z = torch.randn(
            len(captions), *self.latent_shape[1:], device=self.device, dtype=self.dtype
        )
        return {"z": z, "ctx": ctx, "captions": captions}

    def sample_batch(self, n: int) -> dict:
        """Batched rollout input from ``n`` randomly drawn training prompts."""
        captions = [
            self.dataset[self.rng.randrange(len(self.dataset))][1] for _ in range(n)
        ]
        return self.make_batch(captions)

    def _neg_ctx_for(self, batch: int) -> torch.Tensor:
        """Broadcast the single negative prompt embedding across the batch."""
        if self.neg_ctx.shape[0] == batch:
            return self.neg_ctx
        return self.neg_ctx.expand(batch, *self.neg_ctx.shape[1:])

    def _t(self, step_id: int) -> torch.Tensor:
        """Model-facing timestep, always shape ``[1]``.

        Deliberately not expanded to ``[B]``: ``FlowMatchScheduler.step``
        computes ``argmin((self.timesteps - timestep).abs())`` against a
        ``[num_inference_steps]`` tensor, so a ``[B]`` timestep fails to
        broadcast.  Wan's ``t_mod`` (``[1, 6, dim]``) and CacheHead's
        ``TimestepAdaLN`` (``[1, 1, C]``) both broadcast a ``[1]`` timestep
        over the batch correctly, and every sample in a batch shares the same
        rollout step anyway.
        """
        return self.scheduler.timesteps[step_id].reshape(1).to(device=self.device, dtype=self.dtype)

    def _sigma(self, step_id: int) -> torch.Tensor:
        return self.scheduler.sigmas[step_id].to(device=self.device)

    # ---- rollout ---------------------------------------------------------

    @torch.no_grad()
    def _prefix_roll(self, latents, ctx, stop_step: int):
        """Roll steps [0, stop_step) under no_grad; returns (latents, prev_guided)."""
        prev_guided = None
        for k in range(stop_step):
            t = self._t(k)
            if self.schedule.is_full_step(k):
                noise_pred, prev_guided = full_step(self.dit, latents, t, ctx, self.neg_ctx, self.cfg)
            else:
                noise_pred, prev_guided = head_step(
                    self.head, t, prev_guided, self.grid, self.patch,
                    current_latents=latents,
                )
            latents = self.scheduler.step(noise_pred, t, latents)
        return latents, prev_guided

    def _head_step_grad(self, latents, prev_guided, step_j: int):
        """CacheHead update at step j WITH gradients.  Returns
        (v_tokens, noise_pred, x0_G, t_j, sigma_j, head_out).

        ``head_out`` is the residual used directly by current checkpoints.
        Legacy checkpoint scaling is encapsulated by ``CacheHeadConfig``.
        """
        t_j = self._t(step_j)
        sigma_j = self._sigma(step_j)
        noise_pred, v_tokens = head_step(
            self.head, t_j, prev_guided, self.grid, self.patch,
            current_latents=latents,
        )
        head_out = v_tokens - prev_guided
        x0_G = flow_to_x0(latents, noise_pred, sigma_j)
        return v_tokens, noise_pred, x0_G, t_j, sigma_j, head_out

    def _match_sigma(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        # sigma stays float64 for exact flow math; the model-facing timestep is
        # cast to the model dtype like every other timestep in the harness.
        sigma = torch.rand(batch, device=self.device, dtype=torch.float64)
        sigma = sigma * (self.sigma_max - self.sigma_min) + self.sigma_min
        num_train = float(getattr(self.scheduler, "num_train_timesteps", 1000))
        t = (sigma * num_train).to(dtype=self.dtype)
        return sigma, t

    def _generator_x0(self, sample: dict) -> torch.Tensor:
        """no_grad generator x0 prediction at the sampled head step."""
        latents, prev_guided = self._prefix_roll(sample["z"], sample["ctx"], sample["step_j"])
        with torch.no_grad():
            _, _, x0_G, _, _, _ = self._head_step_grad(latents, prev_guided, sample["step_j"])
        return x0_G

    # ---- loss arms -------------------------------------------------------

    def regression(self, sample: dict) -> dict:
        latents, prev_guided = self._prefix_roll(sample["z"], sample["ctx"], sample["step_j"])
        v_tokens, _, x0_G, t_j, _, head_out = self._head_step_grad(
            latents, prev_guided, sample["step_j"]
        )
        with torch.no_grad():
            _, teacher_tokens = full_step(self.dit, latents, t_j, sample["ctx"], self.neg_ctx, self.cfg)
        loss = regression_loss(v_tokens, teacher_tokens, self.reg_loss)
        return {"loss": loss, "x0_G": x0_G, "v_tokens": v_tokens}

    def dmd(self, sample: dict) -> dict:
        latents, prev_guided = self._prefix_roll(sample["z"], sample["ctx"], sample["step_j"])
        v_tokens, _, x0_G, t_j, _, head_out = self._head_step_grad(
            latents, prev_guided, sample["step_j"]
        )
        # Perturb the generated x0 and query teacher + fake-score at the same state.
        sigma, t = self._match_sigma(x0_G.shape[0])
        eps = torch.randn_like(x0_G)
        x_t = forward_diffuse(x0_G, eps, sigma)
        with torch.no_grad():
            v_teacher, _ = full_step(self.dit, x_t, t, sample["ctx"], self.neg_ctx, self.cfg)
            teacher_x0 = flow_to_x0(x_t, v_teacher, sigma)
            v_fake, _ = full_step(self.fake_score, x_t, t, sample["ctx"], self.neg_ctx, self.cfg)
            fake_x0 = flow_to_x0(x_t, v_fake, sigma)
        loss = dmd_loss(x0_G, fake_x0, teacher_x0)
        if self.arm == "dmd_plus_reg":
            with torch.no_grad():
                _, teacher_tokens = full_step(self.dit, latents, t_j, sample["ctx"], self.neg_ctx, self.cfg)
            reg = regression_loss(v_tokens, teacher_tokens, self.reg_loss)
            loss = loss + self.reg_weight * reg
        return {"loss": loss, "x0_G": x0_G, "v_tokens": v_tokens}

    @torch.no_grad()
    def teacher_guided_trajectory(self, batch: dict) -> torch.Tensor:
        """Sample a full-Wan trajectory and return guided tokens ``[B,T,S,C]``.

        Every scheduler update uses the teacher's latent velocity.  The saved
        guided tokens are consequently fixed teacher-forcing inputs/targets;
        the student never changes this trajectory.
        """
        latents, ctx = batch["z"], batch["ctx"]
        n_batch = latents.shape[0]
        neg_ctx = self._neg_ctx_for(n_batch)
        guided_steps = []
        for k in range(self.schedule.num_inference_steps):
            t = self._t(k)
            noise_pred, guided = full_step(self.dit, latents, t, ctx, neg_ctx, self.cfg)
            guided_steps.append(guided.detach())
            latents = self.scheduler.step(noise_pred, t, latents)
        return torch.stack(guided_steps, dim=1)

    @torch.no_grad()
    def reconstruct_teacher_latents(
        self, initial_latents: torch.Tensor, teacher_guided: torch.Tensor
    ) -> torch.Tensor:
        """Rebuild each teacher state ``x_k`` from ``x_0`` and cached velocities.

        Returns ``[B,T,C,F,H,W]`` containing the latent at the *start* of each
        denoising step.  This keeps the persistent cache velocity-only while
        providing exact current-latent conditioning during supervision.
        """
        if initial_latents.shape[0] != teacher_guided.shape[0]:
            raise ValueError(
                "initial_latents and teacher_guided must have the same batch size"
            )
        if tuple(initial_latents.shape[1:]) != tuple(self.latent_shape[1:]):
            raise ValueError(
                f"initial_latents must have shape [B,{','.join(map(str, self.latent_shape[1:]))}], "
                f"got {tuple(initial_latents.shape)}"
            )
        latents = initial_latents.to(device=self.device, dtype=self.dtype)
        starts = []
        for k in range(self.schedule.num_inference_steps):
            starts.append(latents)
            velocity = unpatchify_tokens(teacher_guided[:, k], self.grid, self.patch)
            latents = self.scheduler.step(velocity, self._t(k), latents)
        return torch.stack(starts, dim=1)

    def supervised_teacher_forced(
        self,
        teacher_guided: torch.Tensor,
        initial_latents: torch.Tensor | None = None,
    ) -> dict:
        """MSE between student and teacher velocities on student steps only.

        ``teacher_guided[:, k-1]`` is the student's carry at step ``k`` and
        ``teacher_guided[:, k]`` is its target.  Both are from the fixed full
        teacher rollout, so no student prediction is fed into a later step.
        """
        n_batch = teacher_guided.shape[0]
        expected = (
            n_batch,
            self.schedule.num_inference_steps,
            self.grid[0] * self.grid[1] * self.grid[2],
            WAN_NOISE_TOKEN_CHANNELS,
        )
        if tuple(teacher_guided.shape) != expected:
            raise RuntimeError(
                f"teacher-guided trajectory must have shape {expected}, got "
                f"{tuple(teacher_guided.shape)}"
            )
        head_config = self.distributed.unwrap(self.head).config
        teacher_latents = None
        if head_config.head_variant != "legacy":
            if initial_latents is None:
                raise ValueError(
                    f"head_variant {head_config.head_variant!r} requires deterministic initial_latents"
                )
            teacher_latents = self.reconstruct_teacher_latents(
                initial_latents, teacher_guided
            )

        total = None
        per_step: list[dict] = []
        for k in range(self.schedule.num_inference_steps):
            if self.schedule.is_full_step(k):
                continue
            if k == 0:
                raise RuntimeError("step 1 must be a dense/full-teacher step")
            previous_teacher = teacher_guided[:, k - 1].detach()
            teacher_target = teacher_guided[:, k].detach()
            if teacher_latents is None:
                residual = self.head(previous_teacher, self._t(k), self.grid)
            else:
                latent_tokens = patchify_latents(
                    teacher_latents[:, k], self.grid, self.patch
                )
                residual = self.head(
                    previous_teacher, self._t(k), self.grid,
                    latent_tokens=latent_tokens,
                )
            student = previous_teacher + residual
            step_loss = F.mse_loss(student.float(), teacher_target.float())
            carry_loss = F.mse_loss(previous_teacher.float(), teacher_target.float())
            relative_improvement = 1.0 - step_loss / carry_loss.clamp_min(
                torch.finfo(torch.float32).tiny
            )
            total = step_loss if total is None else total + step_loss
            per_step.append({
                "step": k + 1,
                "loss": float(step_loss.detach().item()),
                "raw_mse": float(step_loss.detach().item()),
                "carry_mse": float(carry_loss.detach().item()),
                "relative_improvement": float(relative_improvement.detach().item()),
            })

        if total is None:
            raise RuntimeError(f"schedule {self.schedule} has no head steps to supervise")
        loss = total / self.schedule.num_head_steps
        return {"loss": loss, "per_step": per_step}

    def supervised_trajectory(self, batch: dict) -> dict:
        """Build an in-memory full-teacher trajectory, then train from it."""
        teacher_guided = self.teacher_guided_trajectory(batch)
        return self.supervised_teacher_forced(teacher_guided, batch["z"])

    # ---- persistent full-teacher trajectory cache ----------------------

    def _trajectory_fingerprint(self) -> str:
        if self._trajectory_fingerprint_value is not None:
            return self._trajectory_fingerprint_value
        negative_hash = hashlib.sha256(
            self.neg_ctx.detach().float().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        spec = {
            "schema": 1,
            "model_id": getattr(
                self.dit, "_cache_head_model_id", "Wan-AI/Wan2.1-T2V-1.3B"
            ),
            "num_inference_steps": self.schedule.num_inference_steps,
            "cfg_scale": self.cfg,
            "patch_size": tuple(self.patch),
            "grid": self.grid,
            "latent_shape": self.latent_shape,
            "dtype": str(self.dtype),
            "trajectory_seed": self.trajectory_seed,
            "negative_context_sha256": negative_hash,
            "timesteps": [float(v) for v in self.scheduler.timesteps.detach().cpu()],
            "sigmas": [float(v) for v in self.scheduler.sigmas.detach().cpu()],
        }
        encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self._trajectory_fingerprint_value = hashlib.sha256(encoded).hexdigest()
        return self._trajectory_fingerprint_value

    @staticmethod
    def _caption_hash(caption: str) -> str:
        return hashlib.sha256(caption.encode("utf-8")).hexdigest()

    def _trajectory_path(self, split: str, caption_id: str, caption: str) -> Path:
        if self.trajectory_dir is None:
            raise RuntimeError("trajectory_dir must be configured before supervised training")
        sample_key = hashlib.sha256(
            f"{caption_id}\0{self._caption_hash(caption)}".encode("utf-8")
        ).hexdigest()
        return self.trajectory_dir / self._trajectory_fingerprint() / split / f"{sample_key}.pt"

    def _sample_seed(self, split: str, caption_id: str) -> int:
        digest = hashlib.sha256(
            f"{self.trajectory_seed}\0{split}\0{caption_id}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") % (2**63 - 1)

    def deterministic_initial_latents(
        self, items: list[tuple[str, str]], *, split: str
    ) -> torch.Tensor:
        """Generate the caption-seeded ``x_0`` shared by caching and replay."""
        latents = []
        for caption_id, _ in items:
            generator = torch.Generator(device="cpu").manual_seed(
                self._sample_seed(split, str(caption_id))
            )
            latents.append(
                torch.randn(self.latent_shape[1:], generator=generator, dtype=torch.float32)
            )
        return torch.stack(latents)

    def _load_teacher_trajectory(
        self, path: Path, *, split: str, caption_id: str, caption: str
    ) -> torch.Tensor:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"could not load teacher trajectory cache {path}: {exc}") from exc
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        expected_metadata = {
            "schema": 1,
            "fingerprint": self._trajectory_fingerprint(),
            "split": split,
            "caption_id": str(caption_id),
            "caption_sha256": self._caption_hash(caption),
            "sample_seed": self._sample_seed(split, str(caption_id)),
        }
        if any(metadata.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(
                f"teacher trajectory metadata mismatch for {path}; remove that cache file "
                "or use a different --trajectory-dir"
            )
        tokens = payload.get("guided_tokens")
        expected_shape = (
            self.schedule.num_inference_steps,
            self.grid[0] * self.grid[1] * self.grid[2],
            WAN_NOISE_TOKEN_CHANNELS,
        )
        if not isinstance(tokens, torch.Tensor) or tuple(tokens.shape) != expected_shape:
            actual = None if not isinstance(tokens, torch.Tensor) else tuple(tokens.shape)
            raise RuntimeError(
                f"teacher trajectory {path} has guided_tokens shape {actual}; "
                f"expected {expected_shape}"
            )
        if not torch.isfinite(tokens.float()).all():
            raise RuntimeError(f"teacher trajectory {path} contains NaN or Inf")
        return tokens.contiguous()

    def _save_teacher_trajectory(
        self,
        path: Path,
        tokens: torch.Tensor,
        *,
        split: str,
        caption_id: str,
        caption: str,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "schema": 1,
                "fingerprint": self._trajectory_fingerprint(),
                "split": split,
                "caption_id": str(caption_id),
                "caption_sha256": self._caption_hash(caption),
                "sample_seed": self._sample_seed(split, str(caption_id)),
            },
            # The guided CFG velocity tokens are the complete training contract;
            # latent states and text embeddings are intentionally not persisted.
            "guided_tokens": tokens.detach().cpu().contiguous(),
        }
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{self.distributed.rank}")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load_teacher_batch(
        self, items: list[tuple[str, str]], *, split: str
    ) -> torch.Tensor:
        """Load fixed trajectories, lazily generating all cache misses in one batch."""
        paths = [self._trajectory_path(split, str(cid), caption) for cid, caption in items]
        missing = [i for i, path in enumerate(paths) if not path.is_file()]
        self.trajectory_cache_hits += len(items) - len(missing)
        self.trajectory_cache_misses += len(missing)

        if missing:
            missing_items = [items[i] for i in missing]
            captions = [caption for _, caption in missing_items]
            batch = {
                "z": self.deterministic_initial_latents(
                    missing_items, split=split
                ).to(device=self.device, dtype=self.dtype),
                "ctx": self.encode_captions(captions),
            }
            guided = self.teacher_guided_trajectory(batch).detach().cpu()
            for local_i, source_i in enumerate(missing):
                caption_id, caption = items[source_i]
                self._save_teacher_trajectory(
                    paths[source_i], guided[local_i], split=split,
                    caption_id=str(caption_id), caption=caption,
                )

        loaded = [
            self._load_teacher_trajectory(
                path, split=split, caption_id=str(caption_id), caption=caption
            )
            for path, (caption_id, caption) in zip(paths, items)
        ]
        return torch.stack(loaded, dim=0)

    def fake_score_update(self, sample: dict) -> torch.Tensor:
        """Denoising loss for the LoRA fake-score on a stop-grad generated sample."""
        x0_G = self._generator_x0(sample)
        sigma, t = self._match_sigma(x0_G.shape[0])
        eps = torch.randn_like(x0_G)
        x_t = forward_diffuse(x0_G, eps, sigma)
        v_fake, _ = full_step(
            self.fake_score,
            x_t,
            t,
            sample["ctx"],
            self.neg_ctx,
            self.cfg,
            use_gradient_checkpointing=True,
        )
        fake_x0 = flow_to_x0(x_t, v_fake, sigma)
        return F.mse_loss(fake_x0, x0_G.detach())

    # ---- memory probe + training loop ------------------------------------

    def memory_probe(self, training_type: str = "dmd") -> tuple[int, int]:
        """Report the peak for the *requested* micro-batch and derive accumulation.

        This reports, it does not search.  ``micro_batch`` is an explicit
        setting so the effective batch is identical across runs and across
        ranks; a micro-batch that does not fit raises here, at startup, with an
        actionable message rather than mid-epoch.
        """
        micro = self.micro_batch
        accum = self.batch_size // micro
        if torch.device(self.device).type == "cpu":
            return micro, accum

        torch.cuda.reset_peak_memory_stats()
        try:
            if training_type == "supervised":
                out = self.supervised_trajectory(self.sample_batch(micro))
            else:
                # Representative of the DMD loop: prefix + teacher query + head backward.
                out = self.regression(self.sample_one())
            out["loss"].backward()
        except torch.cuda.OutOfMemoryError as exc:
            self.head_opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            raise torch.cuda.OutOfMemoryError(
                f"--micro-batch {micro} does not fit on {self.device} for arm {self.arm!r}. "
                f"Lower --micro-batch (keeping --batch-size a multiple of it) to trade "
                f"throughput for memory; the effective batch is unchanged."
            ) from exc
        self.head_opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        if self.distributed.is_main_process:
            print(f"[memory probe] micro-batch={micro}, gradient accumulation={accum} "
                  f"(effective batch {micro * accum} per rank, peak {peak / 1024 ** 3:.1f} GiB)")
        return micro, accum

    def _emit(self, line: str, *, print_line: bool = True) -> None:
        """Append every progress line to disk and optionally print it.

        The file is reopened per line so a crashed or killed run still leaves
        every optimizer update it had already completed on disk.
        """
        if print_line:
            print(line)
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def train(
        self,
        checkpoint_every: int = 1000,
        save_dir: str | None = None,
        log_interval: int = 50,
        wandb_run: Any | None = None,
        training_type: str | None = None,
        heatmap_hook: Callable[[int, Path], Any] | None = None,
        log_path: str | Path | None = None,
    ):
        """Run the loop this arm calls for.

        ``training_type`` defaults to the arm's own loop and exists so tests can
        drive either loop directly; an unrecognized value raises rather than
        silently running neither and saving an untrained checkpoint.
        """
        if training_type is None:
            training_type = training_type_for_arm(self.arm)
        if training_type not in TRAINING_TYPES:
            raise ValueError(
                f"training_type must be one of {TRAINING_TYPES}, got {training_type!r}"
            )

        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            if self.distributed.is_main_process:
                save_dir.mkdir(parents=True, exist_ok=True)
            self.distributed.barrier()
        if training_type == "supervised":
            if self.trajectory_dir is None:
                self.trajectory_dir = (
                    save_dir / "trajectories" if save_dir else Path("cache_head_trajectories")
                )
            self.trajectory_dir.mkdir(parents=True, exist_ok=True)
            self.distributed.barrier()

        _, accum = self.memory_probe(training_type)

        self.logs = []
        self.log_path = Path(log_path) if log_path else None
        if training_type == "supervised":
            self._train_supervised(
                accum=accum, checkpoint_every=checkpoint_every, save_dir=save_dir,
                log_interval=log_interval, wandb_run=wandb_run, heatmap_hook=heatmap_hook,
            )
        else:
            self._train_dmd(
                accum=accum, checkpoint_every=checkpoint_every, save_dir=save_dir,
                log_interval=log_interval, wandb_run=wandb_run,
            )

        if save_dir and self.distributed.is_main_process:
            self.save(save_dir / "cache_head_final.ckpt")
            with open(save_dir / "run_log.json", "w", encoding="utf-8") as fh:
                json.dump(self.logs, fh, indent=2)
        self.distributed.barrier()
        return self.logs

    # ---- supervised: epoch + validation loop -----------------------------

    @torch.no_grad()
    def validate(self) -> dict:
        """Mean supervised loss over the val split, plus per-head-step detail.

        Per-head-step loss is the diagnostic that pairs with the error heat
        map: it says *when* the head drifts, the heat map says *where*.
        """
        if self.val_dataset is None:
            return {}
        was_training = self.head.training
        self.head.eval()
        rng = random.Random(self.seed)
        totals: dict[int, float] = {}
        carry_totals: dict[int, float] = {}
        improvement_totals: dict[int, float] = {}
        overall = 0.0
        n_batches = min(self.val_batches, max(1, len(self.val_dataset) // self.micro_batch))
        for _ in range(n_batches):
            items = [
                self.val_dataset[rng.randrange(len(self.val_dataset))]
                for _ in range(self.micro_batch)
            ]
            teacher_guided = self.load_teacher_batch(items, split="val").to(
                device=self.device, dtype=self.dtype
            )
            initial_latents = self.deterministic_initial_latents(items, split="val")
            out = self.supervised_teacher_forced(teacher_guided, initial_latents)
            overall += float(out["loss"].detach().item())
            for rec in out["per_step"]:
                totals[rec["step"]] = totals.get(rec["step"], 0.0) + rec["loss"]
                carry_totals[rec["step"]] = (
                    carry_totals.get(rec["step"], 0.0) + rec["carry_mse"]
                )
                improvement_totals[rec["step"]] = (
                    improvement_totals.get(rec["step"], 0.0)
                    + rec["relative_improvement"]
                )
        if was_training:
            self.head.train()
        return {
            "val_loss": self.distributed.mean_float(overall / n_batches),
            "val_per_step": {
                k: self.distributed.mean_float(v / n_batches) for k, v in sorted(totals.items())
            },
            "val_raw_mse_per_step": {
                k: self.distributed.mean_float(v / n_batches) for k, v in sorted(totals.items())
            },
            "val_carry_per_step": {
                k: self.distributed.mean_float(v / n_batches)
                for k, v in sorted(carry_totals.items())
            },
            "val_relative_improvement_per_step": {
                k: self.distributed.mean_float(v / n_batches)
                for k, v in sorted(improvement_totals.items())
            },
        }

    def _train_supervised(
        self, *, accum: int, checkpoint_every: int, save_dir: Path | None,
        log_interval: int, wandb_run: Any | None,
        heatmap_hook: Callable[[int, Path], Any] | None = None,
    ) -> None:
        items = list(self.dataset)
        per_step_batch = self.micro_batch * accum
        # Every rank must run the same number of optimizer steps: ranks hold
        # disjoint prompt shards that can differ in length, and a rank with an
        # extra step blocks forever in DDP's gradient all-reduce.
        steps_per_epoch = self.distributed.min_int(len(items) // per_step_batch)
        if steps_per_epoch < 1:
            raise ValueError(
                f"rank {self.distributed.rank} has {len(items)} prompts but needs at least "
                f"{per_step_batch} (--micro-batch x accumulation) for one optimizer step; "
                f"raise --subset or lower --batch-size/--micro-batch"
            )

        global_step = 0
        best_val = float("inf")
        for epoch in range(self.epochs):
            order = list(items)
            random.Random(self.seed + epoch).shuffle(order)
            cursor = 0
            for iteration_in_epoch in range(steps_per_epoch):
                cached_micro_batches = []
                for micro_step in range(accum):
                    chunk = order[cursor:cursor + self.micro_batch]
                    cursor += self.micro_batch
                    cached_micro_batches.append((
                        self.load_teacher_batch(chunk, split="train"),
                        self.deterministic_initial_latents(chunk, split="train"),
                    ))

                iteration = epoch * steps_per_epoch + iteration_in_epoch
                for inner_step in range(self.optimizer_steps_per_iteration):
                    self.head_opt.zero_grad(set_to_none=True)
                    micro_losses = []
                    per_step_acc: dict[int, float] = {}
                    per_step_carry_acc: dict[int, float] = {}
                    per_step_improvement_acc: dict[int, float] = {}
                    for micro_step, (cached, initial_latents) in enumerate(cached_micro_batches):
                        with self.distributed.no_sync(
                            self.head, enabled=micro_step < accum - 1
                        ):
                            teacher_guided = cached.to(device=self.device, dtype=self.dtype)
                            out = self.supervised_teacher_forced(
                                teacher_guided, initial_latents
                            )
                            (out["loss"] / accum).backward()
                            micro_losses.append(float(out["loss"].detach().item()))
                            for rec in out["per_step"]:
                                per_step_acc[rec["step"]] = (
                                    per_step_acc.get(rec["step"], 0.0) + rec["loss"]
                                )
                                per_step_carry_acc[rec["step"]] = (
                                    per_step_carry_acc.get(rec["step"], 0.0)
                                    + rec["carry_mse"]
                                )
                                per_step_improvement_acc[rec["step"]] = (
                                    per_step_improvement_acc.get(rec["step"], 0.0)
                                    + rec["relative_improvement"]
                                )
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.grad_clip)
                    )
                    self.head_opt.step()

                    loss_value = sum(micro_losses) / len(micro_losses)
                    finite = self.distributed.all_true(
                        bool(torch.isfinite(torch.tensor(loss_value)).item())
                    )
                    if not finite:
                        raise RuntimeError(
                            f"non-finite loss at epoch {epoch} iteration "
                            f"{iteration_in_epoch} inner step {inner_step}"
                        )

                    record = {
                        "step": global_step,
                        "iteration": iteration,
                        "inner_step": inner_step,
                        "epoch": epoch,
                        "phase": self.arm,
                        "loss": loss_value,
                        "grad_norm": grad_norm,
                        "fake_score_loss": float("nan"),
                        "per_step": {k: v / accum for k, v in sorted(per_step_acc.items())},
                        "raw_mse_per_step": {
                            k: v / accum for k, v in sorted(per_step_acc.items())
                        },
                        "carry_per_step": {
                            k: v / accum for k, v in sorted(per_step_carry_acc.items())
                        },
                        "relative_improvement_per_step": {
                            k: v / accum
                            for k, v in sorted(per_step_improvement_acc.items())
                        },
                        "finite": finite,
                    }
                    if self.distributed.is_main_process:
                        self.logs.append(record)
                        self._emit(
                            f"[epoch {epoch} iteration {iteration_in_epoch}/{steps_per_epoch} "
                            f"inner {inner_step + 1}/{self.optimizer_steps_per_iteration}] "
                            f"{record['phase']} loss={record['loss']:.4e} "
                            f"grad_norm={record['grad_norm']:.4e}",
                            print_line=global_step % log_interval == 0,
                        )
                        if wandb_run is not None:
                            wandb_run.log(record, step=global_step)

                    if (save_dir and checkpoint_every > 0
                            and self.distributed.is_main_process and global_step > 0
                            and global_step % checkpoint_every == 0):
                        self.save(save_dir / f"cache_head_step-{global_step}.ckpt")
                    global_step += 1

            # --- end of epoch: validation, best checkpoint, heat map ---
            metrics = self.validate()
            if metrics:
                if self.distributed.is_main_process:
                    self.logs.append(
                        {"step": global_step, "epoch": epoch, "phase": "val", **metrics}
                    )
                    self._emit(f"[epoch {epoch}] val_loss={metrics['val_loss']:.4e}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {"val_loss": metrics["val_loss"], "epoch": epoch}, step=global_step
                        )
                    if save_dir and metrics["val_loss"] < best_val:
                        best_val = metrics["val_loss"]
                        self.save(save_dir / "cache_head_best.ckpt")
            if heatmap_hook is not None and save_dir and self.distributed.is_main_process:
                heatmap_hook(epoch, save_dir)
            self.distributed.barrier()

    # ---- DMD: step-based loop --------------------------------------------

    def _train_dmd(
        self, *, accum: int, checkpoint_every: int, save_dir: Path | None,
        log_interval: int, wandb_run: Any | None,
    ) -> None:
        total = self.warmup_steps + self.updates
        for global_step in range(total):
            use_dmd = self.arm in DMD_ARMS and global_step >= self.warmup_steps

            # --- CacheHead update (gradient accumulation) ---
            self.head_opt.zero_grad(set_to_none=True)
            for micro_step in range(accum):
                with self.distributed.no_sync(self.head, enabled=micro_step < accum - 1):
                    sample = self.sample_one()
                    out = self.dmd(sample) if use_dmd else self.regression(sample)
                    (out["loss"] / accum).backward()
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.grad_clip)
            )
            self.head_opt.step()

            # --- fake-score updates (5 per CacheHead update, dmd arms only) ---
            if use_dmd:
                self.fake_score.train()
                fs_grad_norms = []
                for _ in range(5):
                    sample = self.sample_one()
                    fs_loss = self.fake_score_update(sample)
                    self.fake_opt.zero_grad(set_to_none=True)
                    fs_loss.backward()
                    fs_grad_norms.append(float(torch.nn.utils.clip_grad_norm_(
                        self.fake_score_module.lora_parameters(), self.grad_clip)))
                    self.fake_opt.step()
                self.fake_score.eval()
                fake_grad_norm = sum(fs_grad_norms) / len(fs_grad_norms)
            else:
                fs_loss = torch.tensor(float("nan"))
                fake_grad_norm = float("nan")

            finite = self.distributed.all_true(bool(torch.isfinite(out["loss"]).all().item()))
            if not finite:
                raise RuntimeError(f"non-finite loss at step {global_step}")

            record = {
                "step": global_step,
                "phase": "warmup" if not use_dmd else self.arm,
                "loss": float(out["loss"].detach().item()),
                "grad_norm": grad_norm,
                "fake_score_loss": float(fs_loss.detach().item()) if fs_loss.ndim == 0 else float("nan"),
                "fake_grad_norm": fake_grad_norm,
                "finite": finite,
            }
            if self.distributed.is_main_process:
                self.logs.append(record)
                self._emit(
                    f"[{global_step}/{total}] {record['phase']} "
                    f"loss={record['loss']:.4e} "
                    f"grad_norm={record['grad_norm']:.4e} "
                    f"fake={record['fake_score_loss']:.4e}",
                    print_line=global_step % log_interval == 0,
                )
                if wandb_run is not None:
                    wandb_run.log(record, step=global_step)

            if (save_dir and checkpoint_every > 0 and self.distributed.is_main_process and global_step > 0
                    and global_step % checkpoint_every == 0):
                self.save(save_dir / f"cache_head_step-{global_step}.ckpt")

    def save(self, path: str | Path) -> Path:
        head = self.distributed.unwrap(self.head)
        cfg = replace(
            head.config,
            model_id=getattr(self.dit, "_cache_head_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
            schedule=self.schedule,
            cfg_scale=self.cfg,
        )
        return save_cache_head(head, cfg, path)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CacheHead training harness",
        epilog=(
            "Eight-A100 DDP launch: torchrun --standalone --nproc_per_node=8 "
            "cache_head_model_training.py [arguments]"
        ),
    )
    parser.add_argument("--arm", choices=list(TRAINING_ARMS), required=True)
    parser.add_argument(
        "--head-variant", choices=list(HEAD_VARIANTS), default="legacy",
        help="CacheHead architecture for the controlled latent-conditioning ablation",
    )
    parser.add_argument("--captions", required=True, help="MixKit caption JSONL path")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--no-network", action="store_true",
                        help="offline mode: never contact modelscope/HuggingFace/W&B. "
                             "Model files must already sit under "
                             "--model-base-path/<model-id>/ (see --model-base-path)")
    parser.add_argument("--model-base-path", default=None,
                        help="local model root for --no-network; weights are read from "
                             "<path>/<model-id>/... (default: DIFFSYNTH_MODEL_BASE_PATH, "
                             "else ./models)")
    parser.add_argument("--subset", type=int, default=1024, help="train subset (first proof: 1024)")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="effective batch per rank, implemented with gradient accumulation")
    parser.add_argument("--micro-batch", type=int, default=1,
                        help="prompts per forward pass; must divide --batch-size. "
                             "Raise it until the startup memory probe reports an OOM, then "
                             "step back one")
    parser.add_argument("--epochs", type=int, default=1,
                        help="supervised arm: passes over the train prompt split")
    parser.add_argument("--val-subset", type=int, default=128,
                        help="supervised arm: prompts drawn from the held-out val split")
    parser.add_argument("--val-batches", type=int, default=8,
                        help="supervised arm: validation batches per epoch")
    parser.add_argument(
        "--optimizer-steps-per-iteration", type=int, default=5,
        help="supervised arm: optimizer updates over each fixed teacher batch (default: 5)",
    )
    parser.add_argument(
        "--trajectory-dir", default=None,
        help="persistent guided-token teacher cache (default: <save-dir>/trajectories)",
    )
    parser.add_argument("--heatmap-every", type=int, default=0,
                        help="supervised arm: render a per-patch head-vs-teacher error heat map "
                             "every N epochs (0 disables)")
    parser.add_argument("--heatmap-prompts", type=int, default=2,
                        help="prompts in the fixed heat-map batch")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=0.1)
    parser.add_argument("--reg-loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=15)
    parser.add_argument(
        "--full-steps", type=parse_full_step_indices, default=None,
        help="1-indexed anchor (full-Wan / 'dense') step positions, comma-separated, "
             "e.g. '1,2,3,4,5,6,7' (default: the schedule's built-in anchors)",
    )
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None,
                        help="single-process device; torchrun selects cuda:LOCAL_RANK")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--save-dir", default="cache_head_output")
    parser.add_argument(
        "--checkpoint-every", type=int, default=2000,
        help="save CacheHead checkpoints every N optimizer steps (0 disables intermediate saves)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=50,
        help="print every N optimizer steps; disk and W&B record every optimizer step",
    )
    parser.add_argument(
        "--log-dir", default="training_logs",
        help="directory for the plain-text run log; rank 0 writes "
             "<log-dir>/<arm>-<start time>.txt there (pass '' to disable)",
    )
    parser.add_argument(
        "--wandb-project",
        help="enable Weights & Biases logging to this project (rank 0 only)",
    )
    parser.add_argument("--wandb-entity", help="optional Weights & Biases entity/team")
    parser.add_argument(
        "--wandb-run-name",
        help="optional W&B base run name; the launcher appends its start time",
    )
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline"], default="online",
        help="Weights & Biases mode when --wandb-project is set",
    )
    args = parser.parse_args()
    args.run_start_time = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    if args.wandb_run_name:
        args.wandb_run_name = f"{args.wandb_run_name}-{args.run_start_time}"
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.micro_batch < 1:
        parser.error("--micro-batch must be positive")
    if args.batch_size % args.micro_batch != 0:
        parser.error(
            f"--batch-size ({args.batch_size}) must be a multiple of "
            f"--micro-batch ({args.micro_batch})"
        )
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.optimizer_steps_per_iteration < 1:
        parser.error("--optimizer-steps-per-iteration must be positive")
    if args.heatmap_every < 0:
        parser.error("--heatmap-every must be non-negative")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")

    apply_no_network(args)

    distributed = initialize_distributed(args.device)
    try:
        run_training(args, distributed)
    finally:
        distributed.cleanup()


def run_training(args: argparse.Namespace, distributed: DistributedContext) -> None:

    from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
    from diffsynth.models.wan_video_dit import set_to_torch_norm

    device = distributed.device
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Explicit as well as via the environment: skip_download=True short-circuits
    # ModelConfig.require_downloading() regardless of how the env is configured.
    model_kwargs = {"skip_download": True} if args.no_network else {}
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors", **model_kwargs),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", **model_kwargs),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth", **model_kwargs),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="google/umt5-xxl/", **model_kwargs),
    )
    dit = pipe.dit
    set_to_torch_norm([dit])
    dit.eval()
    dit.requires_grad_(False)
    dit._cache_head_model_id = args.model_id
    pipe.text_encoder.eval()
    pipe.text_encoder.requires_grad_(False)

    @torch.no_grad()
    def encode(text: str | list[str]) -> torch.Tensor:
        """Encode one caption or a list of them to ``[B, L, D]``.

        The tokenizer pads to a fixed ``seq_len`` (512) and accepts a list, so
        a batch stacks natively.  Note the per-row indexing when zeroing the
        padding: ``diffsynth``'s own ``encode_prompt`` writes ``emb[:, v:] = 0``
        inside the loop, which zeroes *every* row at *every* sequence length.
        That is invisible at B=1 and silently truncates all but the shortest
        caption at B>1.
        """
        ids, mask = pipe.tokenizer(text, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = pipe.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            emb[i, v:] = 0
        return emb.detach()

    def text_encode(caption: str) -> torch.Tensor:
        return encode(caption)

    def text_encode_batch(captions: list[str]) -> torch.Tensor:
        return encode(list(captions))

    neg_ctx = encode(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    schedule_kwargs = {"full_step_indices": args.full_steps} if args.full_steps else {}
    schedule = CacheHeadSchedule(num_inference_steps=args.num_steps, **schedule_kwargs)
    if args.arm == "supervised" and not schedule.is_full_step(0):
        raise ValueError("supervised teacher forcing requires step 1 in --full-steps")
    if args.arm == "supervised" and schedule.num_head_steps == 0:
        raise ValueError("supervised teacher forcing requires at least one student step")
    config = CacheHeadConfig(
        model_id=args.model_id,
        schedule=schedule,
        cfg_scale=args.cfg,
        head_variant=args.head_variant,
        version=2 if args.head_variant == "legacy" else 3,
    )
    head = CacheHead(config).to(device=device, dtype=dtype)
    # FakeScoreWan deep-copies the whole DiT; only the DMD arms need it, and
    # skipping it frees that memory for a larger --micro-batch.
    if args.arm in DMD_ARMS:
        fake_score = FakeScoreWan(dit, rank=args.lora_rank).to(device=device, dtype=dtype)
        fake_score = distributed.wrap(fake_score)
    else:
        fake_score = None
    head = distributed.wrap(head)

    dataset = PromptDataset(
        args.captions,
        split="train",
        subset=args.subset,
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    val_dataset = None
    if args.arm == "supervised":
        val_dataset = PromptDataset(
            args.captions,
            split="val",
            subset=args.val_subset,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
    if distributed.is_main_process:
        print(f"[DDP] world size={distributed.world_size}, per-rank device={device}")
        print(f"Train prompts per rank: {len(dataset)} "
              f"(local split checksum {prompt_split_checksum([cid for cid, _ in dataset.items])})")
        if val_dataset is not None:
            print(f"Val prompts per rank: {len(val_dataset)}")

    scheduler = pipe.scheduler
    scheduler.set_timesteps(schedule.num_inference_steps, denoising_strength=1.0, shift=5.0)

    z_dim = pipe.vae.model.z_dim
    latent_frames = (args.num_frames - 1) // 4 + 1
    latent_h = args.height // pipe.vae.upsampling_factor
    latent_w = args.width // pipe.vae.upsampling_factor
    latent_shape = (1, z_dim, latent_frames, latent_h, latent_w)
    grid = (latent_frames, latent_h // dit.patch_size[1], latent_w // dit.patch_size[2])

    trainer = CacheHeadTrainer(
        dit=dit, scheduler=scheduler, head=head, fake_score=fake_score,
        text_encode=text_encode, neg_ctx=neg_ctx, dataset=dataset,
        schedule=schedule, cfg_scale=args.cfg, patch_size=dit.patch_size, grid=grid,
        latent_shape=latent_shape, arm=args.arm, reg_loss=args.reg_loss,
        reg_weight=args.reg_weight, lr=args.lr, lora_lr=args.lora_lr,
        batch_size=args.batch_size, micro_batch=args.micro_batch,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps, updates=args.updates,
        epochs=args.epochs,
        optimizer_steps_per_iteration=args.optimizer_steps_per_iteration,
        trajectory_dir=args.trajectory_dir, trajectory_seed=args.seed,
        text_encode_batch=text_encode_batch,
        val_dataset=val_dataset, val_batches=args.val_batches,
        device=device, dtype=dtype, seed=args.seed + distributed.rank,
        distributed=distributed,
    )
    wandb_run = initialize_wandb(args, distributed)
    try:
        trainer.train(
            checkpoint_every=args.checkpoint_every,
            save_dir=args.save_dir,
            log_interval=args.log_interval,
            wandb_run=wandb_run,
            heatmap_hook=build_heatmap_hook(args, trainer, wandb_run),
            log_path=open_run_log(args, distributed),
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def open_run_log(args: argparse.Namespace, distributed: DistributedContext) -> Path | None:
    """Create ``--log-dir`` and start a run log stamped with the launch time.

    Only rank 0 keeps a file: the other ranks never print progress lines, so
    they would only create empty logs.  The header records the exact command so
    a log read months later still says which run produced it.
    """
    if not args.log_dir or not distributed.is_main_process:
        return None
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{args.arm}-{args.run_start_time}.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# CacheHead training -- arm={args.arm} started={args.run_start_time}\n")
        fh.write(f"# world_size={distributed.world_size} device={distributed.device}\n")
        fh.write(f"# command: {' '.join(sys.argv)}\n")
    print(f"[log] run log: {path}")
    return path


def apply_no_network(args: argparse.Namespace) -> None:
    """Resolve every model from local disk and make no outbound requests.

    ``ModelConfig.download_if_necessary`` only calls the hub when
    ``require_downloading()`` is true; with downloads skipped it falls through
    to globbing ``<model-base-path>/<model_id>/<origin_file_pattern>``.  The
    Hugging Face variables cover the tokenizer, which loads through
    ``AutoTokenizer.from_pretrained`` and would otherwise still try to reach the
    hub for a revision check.

    Must run before the deferred ``diffsynth`` / ``transformers`` imports in
    ``run_training``, which is why it is called from ``main``.
    """
    if not args.no_network:
        return
    if getattr(args, "wandb_project", None):
        print("[no-network] disabling W&B")
        args.wandb_project = None
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    if args.model_base_path:
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = args.model_base_path
    # setdefault: never override an offline setup the operator already made.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "./models")
    print(f"[no-network] downloads disabled; resolving models under {base}/<model_id>/")


def build_heatmap_hook(
    args: argparse.Namespace, trainer: CacheHeadTrainer, wandb_run: Any | None
) -> Callable[[int, Path], None] | None:
    """End-of-epoch per-patch error heat map, or ``None`` when disabled.

    Uses a fixed prompt set and a fixed seed on every call so successive epochs
    are directly comparable -- a heat map that moves because the prompts moved
    would say nothing about the head.
    """
    if args.heatmap_every <= 0 or args.arm != "supervised":
        return None

    from cache_head_error_heatmap import (
        collect_step_errors,
        print_prompts,
        render_error_heatmap,
        save_error_arrays,
    )

    source = trainer.val_dataset if trainer.val_dataset is not None else trainer.dataset
    captions = [caption for _, caption in source.items[:args.heatmap_prompts]]
    if not captions:
        return None

    def hook(epoch: int, save_dir: Path) -> None:
        if epoch % args.heatmap_every != 0:
            return
        print_prompts(captions, prefix=f"[epoch {epoch} heatmap]")
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        latents = torch.randn(
            len(captions), *trainer.latent_shape[1:], generator=generator
        ).to(device=trainer.device, dtype=trainer.dtype)
        head = trainer.distributed.unwrap(trainer.head)
        was_training = head.training
        head.eval()
        try:
            result = collect_step_errors(
                dit=trainer.dit, scheduler=trainer.scheduler, head=head,
                schedule=trainer.schedule, cfg_scale=trainer.cfg,
                patch_size=trainer.patch, grid=trainer.grid,
                latents=latents, ctx=trainer.encode_captions(captions),
                neg_ctx=trainer.neg_ctx,
            )
        finally:
            if was_training:
                head.train()
        out_dir = Path(save_dir) / "heatmaps"
        png = render_error_heatmap(
            result, out_dir / f"error_heatmap_epoch-{epoch}.png",
            title=f"CacheHead vs frozen Wan · epoch {epoch} · {len(captions)} prompts",
        )
        save_error_arrays(result, out_dir / f"error_heatmap_epoch-{epoch}.pt")
        print(f"[epoch {epoch}] wrote {png}")
        if wandb_run is not None:
            try:
                import wandb

                wandb_run.log({"error_heatmap": wandb.Image(str(png)), "epoch": epoch})
            except Exception as exc:                      # logging must not kill a run
                print(f"[epoch {epoch}] W&B heat-map upload skipped: {exc}")

    return hook


def initialize_wandb(args: argparse.Namespace, distributed: DistributedContext) -> Any | None:
    """Start one W&B run for a DDP job, or return ``None`` when disabled."""
    if not args.wandb_project:
        return None

    run = None
    error = ""
    if distributed.is_main_process:
        try:
            import wandb

            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                mode=args.wandb_mode,
                config={**vars(args), "world_size": distributed.world_size},
                save_code=False,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

    initialized = distributed.all_true(not error)
    if not initialized:
        if run is not None:
            run.finish(exit_code=1)
        detail = error if distributed.is_main_process else "W&B initialization failed on rank 0"
        raise RuntimeError(
            "could not initialize Weights & Biases; install it with "
            "`pip install wandb` and check your login/configuration. " + detail
        )
    return run


if __name__ == "__main__":
    main()
