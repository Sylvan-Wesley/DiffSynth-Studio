"""
CacheHead training harness + loss study.

For each training sample the harness
  1. samples noise, a prompt, and one of the ten head-step indices,
  2. rolls the scheduled prefix under ``no_grad`` to the hybrid latent ``x_i``,
  3. runs the chosen CacheHead update with gradients to produce ``x_{i+1}``,
  4. queries frozen full Wan at the same hybrid state (never an all-full
     reference trajectory),
  5. updates CacheHead only.

Arms:
    carry_previous       no learned head
    residual_regression  Huber/MSE to frozen-Wan velocity at the hybrid state
    dmd                  Strict DMD after a shared regression warm-up
    dmd_plus_reg         DMD + regression (sweep ``--reg-weight`` 0.03/0.1/0.3)

Strict DMD uses a training-only LoRA-Wan fake-score estimator
(``fake_score_wan.py``) on top of frozen Wan's score prediction; one CacheHead
update alternates with four fake-score updates.  The fake-score is discarded
after training; only CacheHead weights + config are exported.

Loss conventions follow ``diffsynth/diffusion/dmd2.py``:
    flow_to_x0(latents, flow, sigma) = latents - sigma*flow
    weight w = 1/(|x0 - teacher_x0|.abs().mean + 1e-6)
    L_DMD = 0.5*|x0 - sg[x0 - w*(fake_x0 - teacher_x0)]|^2
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from cache_head_model import (
    CacheHead,
    CacheHeadConfig,
    CacheHeadSchedule,
    save_cache_head,
    unpatchify_tokens,
)
from cache_head_model_inference import full_step, head_step
from cache_head_ddp import DistributedContext, initialize_distributed
from fake_score_wan import FakeScoreWan


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
        fake_score: FakeScoreWan,
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
        grad_clip: float = 1.0,
        warmup_steps: int = 2000,
        updates: int = 10000,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        sigma_min: float = 0.001,
        sigma_max: float = 0.999,
        seed: int = 0,
        distributed: DistributedContext | None = None,
    ):
        if arm not in ("residual_regression", "dmd", "dmd_plus_reg"):
            raise ValueError(f"arm must be a training arm, got {arm!r} (carry_previous is not trained)")
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
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps
        self.updates = updates
        self.device = device
        self.dtype = dtype
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.seed = seed
        self.distributed = distributed or DistributedContext(
            rank=0, local_rank=0, world_size=1, device=torch.device(device)
        )

        self.rng = random.Random(seed)
        torch.manual_seed(seed)

        # Freeze the teacher; only CacheHead trains in the production optimizer.
        self.dit.eval()
        self.dit.requires_grad_(False)
        assert all(not p.requires_grad for p in self.dit.parameters()), "teacher Wan must be frozen"
        self.head.train()
        self.head_opt = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-2)
        self.fake_score_module = self.distributed.unwrap(self.fake_score)
        self.fake_opt = torch.optim.AdamW(
            list(self.fake_score_module.lora_parameters()), lr=lora_lr, weight_decay=1e-2
        )
        self.fake_score.eval()

        self.logs: list[dict] = []

    # ---- sampling --------------------------------------------------------

    def sample_one(self) -> dict:
        _, caption = self.dataset[self.rng.randrange(len(self.dataset))]
        ctx = self.text_encode(caption)
        z = torch.randn(*self.latent_shape, device=self.device, dtype=self.dtype)
        # 1-indexed head index -> 0-indexed progress id.
        step_j = self.rng.choice(self.schedule.head_step_indices) - 1
        return {"z": z, "ctx": ctx, "step_j": step_j}

    def _t(self, step_id: int) -> torch.Tensor:
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
                noise_pred, prev_guided = head_step(self.head, t, prev_guided, self.grid, self.patch)
            latents = self.scheduler.step(noise_pred, t, latents)
        return latents, prev_guided

    def _head_step_grad(self, latents, prev_guided, step_j: int):
        """CacheHead update at step j WITH gradients.  Returns
        (v_tokens, noise_pred, x0_G, t_j, sigma_j)."""
        t_j = self._t(step_j)
        sigma_j = self._sigma(step_j)
        v_tokens = prev_guided + self.head(prev_guided, t_j, self.grid)
        noise_pred = unpatchify_tokens(v_tokens, self.grid, self.patch)
        x0_G = flow_to_x0(latents, noise_pred, sigma_j)
        return v_tokens, noise_pred, x0_G, t_j, sigma_j

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
            _, _, x0_G, _, _ = self._head_step_grad(latents, prev_guided, sample["step_j"])
        return x0_G

    # ---- loss arms -------------------------------------------------------

    def regression(self, sample: dict) -> dict:
        latents, prev_guided = self._prefix_roll(sample["z"], sample["ctx"], sample["step_j"])
        v_tokens, _, x0_G, t_j, _ = self._head_step_grad(
            latents, prev_guided, sample["step_j"]
        )
        with torch.no_grad():
            _, teacher_tokens = full_step(self.dit, latents, t_j, sample["ctx"], self.neg_ctx, self.cfg)
        loss = regression_loss(v_tokens, teacher_tokens, self.reg_loss)
        return {"loss": loss, "x0_G": x0_G, "v_tokens": v_tokens}

    def dmd(self, sample: dict) -> dict:
        latents, prev_guided = self._prefix_roll(sample["z"], sample["ctx"], sample["step_j"])
        v_tokens, _, x0_G, t_j, _ = self._head_step_grad(
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

    def memory_probe(self) -> tuple[int, int]:
        """Report the B=1 peak and configure true per-rank accumulation.

        ``sample_one`` produces one sample, so this harness has no larger
        micro-batch to select dynamically.  ``batch_size`` is therefore the
        exact number of B=1 accumulation steps on every rank.
        """
        if torch.device(self.device).type == "cpu":
            return 1, self.distributed.max_int(self.batch_size)
        torch.cuda.reset_peak_memory_stats()
        sample = self.sample_one()
        out = self.regression(sample)  # representative: prefix + teacher query + head backward
        out["loss"].backward()
        self.head_opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated()
        micro = 1
        accum = self.distributed.max_int(self.batch_size)
        torch.cuda.reset_peak_memory_stats()
        if self.distributed.is_main_process:
            print(f"[memory probe] micro-batch={micro}, gradient accumulation={accum} "
                  f"(effective batch {micro * accum} per rank, peak {peak / 1024 ** 3:.1f} GiB)")
        return micro, accum

    def train(
        self,
        checkpoint_every: int = 1000,
        save_dir: str | None = None,
        log_interval: int = 50,
        wandb_run: Any | None = None,
    ):
        _, accum = self.memory_probe()
        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            if self.distributed.is_main_process:
                save_dir.mkdir(parents=True, exist_ok=True)
            self.distributed.barrier()

        total = self.warmup_steps + self.updates
        self.logs = []
        for global_step in range(total):
            use_dmd = self.arm in ("dmd", "dmd_plus_reg") and global_step >= self.warmup_steps

            # --- CacheHead update (gradient accumulation) ---
            self.head_opt.zero_grad(set_to_none=True)
            for micro_step in range(accum):
                with self.distributed.no_sync(self.head, enabled=micro_step < accum - 1):
                    sample = self.sample_one()
                    out = self.dmd(sample) if use_dmd else self.regression(sample)
                    (out["loss"] / accum).backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.grad_clip)
            self.head_opt.step()

            # --- fake-score updates (5 per CacheHead update, dmd arms only) ---
            if use_dmd:
                self.fake_score.train()
                for _ in range(5):
                    sample = self.sample_one()
                    fs_loss = self.fake_score_update(sample)
                    self.fake_opt.zero_grad(set_to_none=True)
                    fs_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.fake_score_module.lora_parameters(), self.grad_clip)
                    self.fake_opt.step()
                self.fake_score.eval()
            else:
                fs_loss = torch.tensor(float("nan"))

            finite = self.distributed.all_true(bool(torch.isfinite(out["loss"]).all().item()))
            if not finite:
                raise RuntimeError(f"non-finite loss at step {global_step}")

            record = {
                "step": global_step,
                "phase": "warmup" if not use_dmd else self.arm,
                "loss": float(out["loss"].detach().item()),
                "fake_score_loss": float(fs_loss.detach().item()) if fs_loss.ndim == 0 else float("nan"),
                "finite": finite,
            }
            if self.distributed.is_main_process:
                self.logs.append(record)
                if global_step % log_interval == 0:
                    print(f"[{global_step}/{total}] {record['phase']} loss={record['loss']:.4e} "
                          f"fake={record['fake_score_loss']:.4e}")
                    if wandb_run is not None:
                        wandb_run.log(record, step=global_step)

            if (save_dir and checkpoint_every > 0 and self.distributed.is_main_process and global_step > 0
                    and global_step % checkpoint_every == 0):
                self.save(save_dir / f"cache_head_step-{global_step}.ckpt")

        if save_dir and self.distributed.is_main_process:
            self.save(save_dir / "cache_head_final.ckpt")
            with open(save_dir / "run_log.json", "w", encoding="utf-8") as fh:
                json.dump(self.logs, fh, indent=2)
        self.distributed.barrier()
        return self.logs

    def save(self, path: str | Path) -> Path:
        cfg = CacheHeadConfig(
            model_id=getattr(self.dit, "_cache_head_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
            schedule=self.schedule,
            cfg_scale=self.cfg,
        )
        return save_cache_head(self.distributed.unwrap(self.head), cfg, path)


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
    parser.add_argument("--arm", choices=["residual_regression", "dmd", "dmd_plus_reg"], required=True)
    parser.add_argument("--captions", required=True, help="MixKit caption JSONL path")
    parser.add_argument("--model-id", default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--subset", type=int, default=1024, help="train subset (first proof: 1024)")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--updates", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="effective batch per rank, implemented with gradient accumulation")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--reg-weight", type=float, default=0.1)
    parser.add_argument("--reg-loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--num-steps", type=int, default=15)
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
        help="print and, when enabled, send W&B metrics every N optimizer steps",
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
    args.wandb_start_time = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    if args.wandb_run_name:
        args.wandb_run_name = f"{args.wandb_run_name}-{args.wandb_start_time}"
    if args.checkpoint_every < 0:
        parser.error("--checkpoint-every must be non-negative")
    if args.log_interval <= 0:
        parser.error("--log-interval must be positive")

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

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[
            ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id=args.model_id, origin_file_pattern="Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(model_id=args.model_id, origin_file_pattern="google/umt5-xxl/"),
    )
    dit = pipe.dit
    set_to_torch_norm([dit])
    dit.eval()
    dit.requires_grad_(False)
    dit._cache_head_model_id = args.model_id
    pipe.text_encoder.eval()
    pipe.text_encoder.requires_grad_(False)

    @torch.no_grad()
    def encode(text: str) -> torch.Tensor:
        ids, mask = pipe.tokenizer(text, return_mask=True, add_special_tokens=True)
        ids = ids.to(pipe.device)
        mask = mask.to(pipe.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        emb = pipe.text_encoder(ids, mask)
        for v in seq_lens:
            emb[:, v:] = 0
        return emb.detach()

    def text_encode(caption: str) -> torch.Tensor:
        return encode(caption)

    neg_ctx = encode(
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，"
        "丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        "杂乱的背景，三条腿，背景人很多，倒着走"
    )

    schedule = CacheHeadSchedule(num_inference_steps=args.num_steps, full_step_indices=(1, 2, 6, 10, 14))
    config = CacheHeadConfig(model_id=args.model_id, schedule=schedule, cfg_scale=args.cfg)
    head = CacheHead(config).to(device=device, dtype=dtype)
    fake_score = FakeScoreWan(dit, rank=args.lora_rank).to(device=device, dtype=dtype)
    head = distributed.wrap(head)
    fake_score = distributed.wrap(fake_score)

    dataset = PromptDataset(
        args.captions,
        split="train",
        subset=args.subset,
        rank=distributed.rank,
        world_size=distributed.world_size,
    )
    if distributed.is_main_process:
        print(f"[DDP] world size={distributed.world_size}, per-rank device={device}")
        print(f"Train prompts per rank: {len(dataset)} "
              f"(local split checksum {prompt_split_checksum([cid for cid, _ in dataset.items])})")

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
        batch_size=args.batch_size, grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps, updates=args.updates,
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
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


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
