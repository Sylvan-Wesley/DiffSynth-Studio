"""
CacheHead: a lightweight residual network that replaces eight of fifteen Wan2.1
denoising calls in a hybrid sampler.

Wan stays frozen.  The default 15-step schedule runs the full Wan model (with
positive/negative CFG) at the 1-indexed anchor steps [1, 2, 3, 4, 5, 6, 7];
the other eight steps use CacheHead.  At a head step the CacheHead predicts a
residual on top of the nearest preceding guided noise-token prediction:

    v_hat_i = v_{i-1} + r_phi(v_{i-1}, t_i)

    v_{i-1} is exactly the nearest preceding guided noise-token prediction
    (refresh from full steps, propagated through consecutive head steps).
    The deployed head sees only those tokens and the current Wan timestep.

Legacy architecture (lightweight token-grid residual network):

RMSNorm -> timestep AdaLN -> channel MLP -> depthwise 3D token-grid mixer
    -> timestep AdaLN -> zero-initialized output projection.

Version-3 latent-conditioned variants losslessly patchify the current raw VAE
latent and fuse it with the previous velocity tokens.  The residual variants
then use one or two pre-norm AdaLN-gated residual MLP/mixer blocks, retaining
an unnormalized fused-feature identity path.

Fresh heads emit exactly zero residual and therefore reproduce
``carry_previous`` before training.  New checkpoints add the head residual
directly; legacy checkpoints retain their historical 0.1 residual scale.

The checkpoint stores the head weights plus the schedule, head architecture,
and CFG scale so inference is self-describing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange


# ═══════════════════════════════════════════════════════════════
# Schedule configuration
# ═══════════════════════════════════════════════════════════════

# Noise-token channel width produced by Wan's DiT head for Wan2.1.
# C_tok = out_dim * prod(patch_size) = 16 * (1 * 2 * 2) = 64.
WAN_NOISE_TOKEN_CHANNELS = 64
HEAD_VARIANTS = (
    "legacy",
    "latent_fusion",
    "latent_residual",
    "latent_residual_deep",
)


@dataclass(frozen=True)
class CacheHeadSchedule:
    """Configuration-only schedule; experiments place anchors without touching
    model code.  Indices are 1-indexed like the spec.  ``is_full_step`` is
    called with a 0-indexed progress id."""

    num_inference_steps: int = 15
    full_step_indices: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)

    def __post_init__(self) -> None:
        # Normalize regardless of construction source (e.g. a list deserialized
        # from a checkpoint) so equality and hashing are stable.
        object.__setattr__(self, "full_step_indices", tuple(self.full_step_indices))
        self.validate()

    def validate(self) -> None:
        n = self.num_inference_steps
        if n < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {n}")
        idx = self.full_step_indices
        if len(idx) == 0:
            raise ValueError("full_step_indices must be non-empty")
        if len(set(idx)) != len(idx):
            raise ValueError(f"full_step_indices must be unique, got {idx}")
        if any(i < 1 or i > n for i in idx):
            raise ValueError(f"full_step_indices out of range for {n} steps: {idx}")
        if not all(a < b for a, b in zip(idx, idx[1:])):
            raise ValueError(f"full_step_indices must be sorted ascending: {idx}")

    @property
    def head_step_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(1, self.num_inference_steps + 1) if i not in self.full_step_indices)

    @property
    def num_full_steps(self) -> int:
        return len(self.full_step_indices)

    @property
    def num_head_steps(self) -> int:
        return self.num_inference_steps - self.num_full_steps

    def is_full_step(self, progress_id: int) -> bool:
        """progress_id is 0-indexed; the schedule is declared 1-indexed."""
        return (progress_id + 1) in self.full_step_indices

    def is_head_step(self, progress_id: int) -> bool:
        return not self.is_full_step(progress_id)


def parse_full_step_indices(spec: str) -> tuple[int, ...]:
    """Parse a comma-separated 1-indexed anchor/dense-step list, e.g. "1,2,6,10,14".

    Only converts and shapes the input; ``CacheHeadSchedule.validate()`` (run
    from ``__post_init__``) is the single source of truth for range,
    uniqueness, and ordering, so this stays a dumb splitter shared by both the
    training and inference CLIs.
    """
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise ValueError(f"must be a non-empty comma-separated list of ints, got {spec!r}")
    try:
        return tuple(int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"must be a comma-separated list of ints, got {spec!r}") from exc


@dataclass(frozen=True)
class CacheHeadConfig:
    """Everything needed to rebuild a deployed head from a checkpoint."""

    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    schedule: CacheHeadSchedule = field(default_factory=CacheHeadSchedule)
    cfg_scale: float = 5.0
    # Head architecture knobs.
    token_channels: int = WAN_NOISE_TOKEN_CHANNELS
    hidden_factor: int = 2          # channel MLP expansion (small)
    freq_dim: int = 256             # sinusoidal timestep embedding width
    mixer_kernel_size: int = 3      # depthwise 3D token-grid mixer kernel
    adaln_dropout: float = 0.0
    residual_scale: float = 1.0
    head_variant: str = "legacy"
    version: int = 2

    def __post_init__(self) -> None:
        if self.token_channels < 1:
            raise ValueError(f"token_channels must be >= 1, got {self.token_channels}")
        if self.cfg_scale < 1.0:
            raise ValueError(f"cfg_scale must be >= 1.0 (CFG is distilled into the head), got {self.cfg_scale}")
        if self.residual_scale <= 0:
            raise ValueError(f"residual_scale must be positive, got {self.residual_scale}")
        if self.head_variant not in HEAD_VARIANTS:
            raise ValueError(
                f"head_variant must be one of {HEAD_VARIANTS}, got {self.head_variant!r}"
            )
        if self.version not in (1, 2, 3):
            raise ValueError(f"version must be 1, 2, or 3, got {self.version}")
        if self.head_variant != "legacy" and self.version != 3:
            raise ValueError(
                f"head_variant {self.head_variant!r} requires checkpoint version 3"
            )
        self.schedule.validate()


# ═══════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════

def sinusoidal_embedding_1d(dim: int, timestep: torch.Tensor) -> torch.Tensor:
    """Standard sinusoidal time embedding; matches the diffusion convention
    where ``timestep`` is a non-negative real (Wan uses sigma * 1000)."""
    half_dim = dim // 2
    exponent = -math.log(10_000.0) * torch.arange(half_dim, device=timestep.device, dtype=torch.float32)
    freqs = torch.exp(exponent / max(half_dim, 1))
    args = timestep.float().unsqueeze(-1) * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class RMSNorm(nn.Module):
    """Non-affine RMSNorm over the last axis (matches Wan's head norm style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(dtype)


class TimestepAdaLN(nn.Module):
    """Per-channel adaptive scale+shift modulated by the current Wan timestep.

    ``x -> x * (1 + scale(t)) + shift(t)``.  The identity bias of 1 on scale
    keeps the layer output near its input scale for small embedding values.
    """

    def __init__(self, dim: int, freq_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.net = nn.Sequential(
            nn.Linear(freq_dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 2 * dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_embedding_1d(self.freq_dim, timestep)  # [B, freq_dim], float32
        # The embedding is built in float32 for precision, but the projection
        # weights follow the model dtype (bf16 under --precision bf16) and
        # F.linear requires both operands to match -- otherwise this raises
        # "mat1 and mat2 must have the same dtype".  Wan's own DiT casts at the
        # same point: sinusoidal_embedding_1d(...).to(x.dtype).
        emb = emb.to(self.net[0].weight.dtype)
        scale, shift = self.net(emb).chunk(2, dim=-1)  # each [B, C]
        scale = self.dropout(scale).unsqueeze(1)
        shift = self.dropout(shift).unsqueeze(1)
        return x * (1.0 + scale) + shift


class TimestepAdaLNGate(nn.Module):
    """AdaLN modulation for a residual branch, including a learned gate.

    The final modulation layer starts with scale/shift exactly zero and gate
    exactly one.  Consequently a residual block initially behaves like a
    conventional pre-norm residual block while retaining the raw, unnormalized
    identity stream.
    """

    def __init__(self, dim: int, freq_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.net = nn.Sequential(
            nn.Linear(freq_dim, 2 * dim),
            nn.SiLU(),
            nn.Linear(2 * dim, 3 * dim),
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.zero_()
            self.net[-1].bias[2 * dim:].fill_(1.0)

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = sinusoidal_embedding_1d(self.freq_dim, timestep)
        emb = emb.to(self.net[0].weight.dtype)
        scale, shift, gate = self.net(emb).chunk(3, dim=-1)
        scale = self.dropout(scale).unsqueeze(1)
        shift = self.dropout(shift).unsqueeze(1)
        gate = self.dropout(gate).unsqueeze(1)
        return x * (1.0 + scale) + shift, gate


def _normalize_timestep(timestep: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Accept a scalar, [1], or [B] timestep and return a [B] float32 tensor."""
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor(float(timestep), device=device, dtype=torch.float32)
    timestep = timestep.to(device=device)
    return timestep.reshape(-1).float()


# ═══════════════════════════════════════════════════════════════
# CacheHead network
# ═══════════════════════════════════════════════════════════════

class ResidualAdaLNBlock(nn.Module):
    """Pre-norm MLP and local token-grid mixer with raw identity paths."""

    def __init__(self, config: CacheHeadConfig):
        super().__init__()
        dim = config.token_channels
        hidden = dim * config.hidden_factor
        self.mlp_norm = RMSNorm(dim)
        self.mlp_adaln = TimestepAdaLNGate(
            dim, freq_dim=config.freq_dim, dropout=config.adaln_dropout
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.mixer_norm = RMSNorm(dim)
        self.mixer_adaln = TimestepAdaLNGate(
            dim, freq_dim=config.freq_dim, dropout=config.adaln_dropout
        )
        k = config.mixer_kernel_size
        self.mixer = nn.Conv3d(
            dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=False
        )

    def forward(
        self, x: torch.Tensor, timestep: torch.Tensor, grid: tuple[int, int, int]
    ) -> torch.Tensor:
        mlp_input, mlp_gate = self.mlp_adaln(self.mlp_norm(x), timestep)
        x = x + mlp_gate * self.mlp(mlp_input)

        mixer_input, mixer_gate = self.mixer_adaln(self.mixer_norm(x), timestep)
        f, h, w = grid
        mixed = rearrange(mixer_input, "b (f h w) c -> b c f h w", f=f, h=h, w=w)
        mixed = self.mixer(mixed)
        mixed = rearrange(mixed, "b c f h w -> b (f h w) c")
        return x + mixer_gate * mixed

class CacheHead(nn.Module):
    """Lightweight token-grid residual network.

    Operates on Wan noise tokens ``[B, S, C]`` (S = f*h*w token grid, C = 64)
    and, for version-3 variants, current raw-latent tokens of the same shape.
    Returns a residual
    ``[B, S, C]`` which the sampler adds to the nearest preceding guided
    noise-token prediction.

    The output projection is zero-initialized by default so a fresh head is the
    exact ``carry_previous`` baseline.
    """

    def __init__(self, config: CacheHeadConfig, *, zero_init_out_proj: bool = True):
        super().__init__()
        self.config = config
        dim = config.token_channels
        hidden = dim * config.hidden_factor

        if config.head_variant == "legacy":
            # Keep the version-1/2 module names and graph unchanged so old
            # state dicts load exactly and produce bit-identical outputs.
            self.norm = RMSNorm(dim)
            self.adaln = TimestepAdaLN(
                dim, freq_dim=config.freq_dim, dropout=config.adaln_dropout
            )
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden),
                nn.GELU(),
                nn.Linear(hidden, dim),
            )
            k = config.mixer_kernel_size
            self.mixer = nn.Conv3d(
                dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=False
            )
            self.adaln2 = TimestepAdaLN(
                dim, freq_dim=config.freq_dim, dropout=config.adaln_dropout
            )
        else:
            self.fusion = nn.Linear(2 * dim, dim)
            if config.head_variant == "latent_fusion":
                self.norm = RMSNorm(dim)
                self.adaln = TimestepAdaLN(
                    dim, freq_dim=config.freq_dim, dropout=config.adaln_dropout
                )
                self.mlp = nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, dim),
                )
                k = config.mixer_kernel_size
                self.mixer = nn.Conv3d(
                    dim, dim, kernel_size=k, padding=k // 2, groups=dim, bias=False
                )
                self.adaln2 = TimestepAdaLN(
                    dim, freq_dim=config.freq_dim, dropout=config.adaln_dropout
                )
            else:
                depth = 2 if config.head_variant == "latent_residual_deep" else 1
                self.blocks = nn.ModuleList(
                    ResidualAdaLNBlock(config) for _ in range(depth)
                )
        #  this is due to the observation that different timestep have different magnitude of residual
        self.out_proj = nn.Linear(dim, dim, bias=False)
        # Zero init -> residual == 0 -> carry_previous.  The opt-out remains
        # useful for diagnostics, but all production construction uses zero.
        if zero_init_out_proj:
            nn.init.zeros_(self.out_proj.weight)
        else:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        timestep: torch.Tensor,
        grid: tuple[int, int, int],
        latent_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``tokens`` [B, S, C]; ``grid`` = (f, h, w) token-grid dims; returns [B, S, C]."""
        S = tokens.shape[1]
        f, h, w = grid
        if f * h * w != S:
            raise ValueError(f"grid {grid} product {f*h*w} != token count {S}")
        device = tokens.device
        t = _normalize_timestep(timestep, device)

        if self.config.head_variant == "legacy":
            x = tokens
        else:
            if latent_tokens is None:
                raise ValueError(
                    f"head_variant {self.config.head_variant!r} requires latent_tokens [B,S,C]"
                )
            if latent_tokens.shape != tokens.shape:
                raise ValueError(
                    f"latent_tokens must match velocity-token shape {tuple(tokens.shape)}, "
                    f"got {tuple(latent_tokens.shape)}"
                )
            if latent_tokens.device != tokens.device or latent_tokens.dtype != tokens.dtype:
                raise ValueError("latent_tokens must have the same device and dtype as tokens")
            x = self.fusion(torch.cat((tokens, latent_tokens), dim=-1))

        if self.config.head_variant in ("legacy", "latent_fusion"):
            x = self.norm(x)
            x = self.adaln(x, t)
            x = self.mlp(x)
            x = rearrange(x, "b (f h w) c -> b c f h w", f=f, h=h, w=w)
            x = self.mixer(x)
            x = rearrange(x, "b c f h w -> b (f h w) c")
            x = self.adaln2(x, t)
        else:
            for block in self.blocks:
                x = block(x, t, grid)
        return self.out_proj(x) * self.config.residual_scale


# The Strict-DMD fake-score estimator is a *LoRA Wan* (a frozen Wan DiT clone
# with a trainable low-rank adapter), defined in ``fake_score_wan.py``.  It is a
# training-only auxiliary model that is never exported.


# ═══════════════════════════════════════════════════════════════
# Token <-> latent velocity helpers
# ═══════════════════════════════════════════════════════════════

def token_grid(latent_frames: int, latent_h: int, latent_w: int, patch_size) -> tuple[int, int, int]:
    """Token-grid dims (f, h, w) after patchifying a latent of shape
    [B, C, latent_frames, latent_h, latent_w] with Wan's patch_size."""
    return (latent_frames, latent_h // patch_size[1], latent_w // patch_size[2])


def unpatchify_tokens(tokens: torch.Tensor, grid: tuple[int, int, int], patch_size) -> torch.Tensor:
    """Convert noise tokens [B, S, C] back to a latent velocity [B, c, f, h, w].

    Mirrors ``WanModel.unpatchify`` (``rearrange('b (f h w) (x y z c) -> b c (f x) (h y) (w z)')``).
    ``tokens`` must be ``[B, S, C]`` with ``C = c * prod(patch_size)``.
    """
    f, h, w = grid
    x, y, z = patch_size
    return rearrange(
        tokens,
        "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
        f=f, h=h, w=w, x=x, y=y, z=z,
    )


def patchify_latents(latents: torch.Tensor, grid: tuple[int, int, int], patch_size) -> torch.Tensor:
    """Lossless inverse of :func:`unpatchify_tokens` for raw Wan latents."""
    if latents.ndim != 5:
        raise ValueError(f"latents must have shape [B,C,F,H,W], got {tuple(latents.shape)}")
    f, h, w = grid
    x, y, z = patch_size
    expected_spatial = (f * x, h * y, w * z)
    if tuple(latents.shape[2:]) != expected_spatial:
        raise ValueError(
            f"latent spatial shape {tuple(latents.shape[2:])} does not match grid {grid} "
            f"and patch_size {tuple(patch_size)} (expected {expected_spatial})"
        )
    return rearrange(
        latents,
        "b c (f x) (h y) (w z) -> b (f h w) (x y z c)",
        f=f, h=h, w=w, x=x, y=y, z=z,
    )


# ═══════════════════════════════════════════════════════════════
# Checkpoint I/O
# ═══════════════════════════════════════════════════════════════

def _deep_asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _deep_asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [_deep_asdict(v) for v in obj]
    return obj


def save_cache_head(head: nn.Module, config: CacheHeadConfig, path: str | Path) -> Path:
    """Save head weights plus schedule, head architecture, and CFG scale."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_version = 3 if config.head_variant != "legacy" else config.version
    if checkpoint_version not in (1, 2, 3):
        raise ValueError(f"Unsupported CacheHead checkpoint version: {checkpoint_version}")
    stored_config = config if config.version == checkpoint_version else CacheHeadConfig(
        **{**_deep_asdict(config), "schedule": config.schedule, "version": checkpoint_version}
    )
    payload = {
        "version": checkpoint_version,
        "config": _deep_asdict(stored_config),
        "model_state_dict": {k: v.cpu() for k, v in head.state_dict().items()},
    }
    torch.save(payload, path)
    return path


def load_cache_head(
    path: str | Path,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[CacheHead, CacheHeadConfig]:
    """Rebuild a CacheHead and its config from a checkpoint.

    ``dtype`` must match the pipeline the head will run inside.  ``CacheHead``
    is constructed in float32 and ``load_state_dict`` copies *into* those
    parameters, so a bf16 checkpoint silently comes back as float32; the head
    then emits float32 tokens, the scheduler promotes the latents to float32,
    and the next full step hands float32 activations to a bf16 Wan.  Pass the
    pipeline dtype to keep the two in step.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CacheHead checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint_version = payload.get("version")
    if checkpoint_version not in (1, 2, 3):
        raise ValueError(f"Unsupported CacheHead checkpoint version: {payload.get('version')}")
    cfg = dict(payload["config"])
    # Version-1 checkpoints were trained and deployed with an external /10 in
    # head_step().  Move that behavior into the self-describing config so the
    # new direct-add call sites preserve old checkpoints exactly.
    if checkpoint_version == 1:
        cfg.setdefault("residual_scale", 0.1)
    if checkpoint_version in (1, 2):
        cfg.setdefault("head_variant", "legacy")
    elif "head_variant" not in cfg:
        raise ValueError("Version-3 CacheHead checkpoint is missing config.head_variant")
    cfg["version"] = checkpoint_version
    schedule = CacheHeadSchedule(**cfg["schedule"])
    config = CacheHeadConfig(
        schedule=schedule, **{k: v for k, v in cfg.items() if k != "schedule"}
    )
    head = CacheHead(config)
    head.load_state_dict(payload["model_state_dict"])
    head.to(device=device) if dtype is None else head.to(device=device, dtype=dtype)
    head.eval()
    return head, config
