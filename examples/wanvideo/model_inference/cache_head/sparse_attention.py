"""Sparse self-attention for the CacheHead student DiT.

Wan's ``SelfAttention`` routes every attention call through a single
parameter-free submodule, ``AttentionModule`` (``diffsynth/models/wan_video_dit.py``),
held as ``block.self_attn.attn``.  Because it owns no parameters, swapping it for
:class:`SparseAttentionModule` leaves ``state_dict()`` byte-identical -- the
student inherits the teacher's weights unchanged and only the attention *pattern*
differs.  The repo already uses this swap idiom in ``enable_usp``
(``diffsynth/pipelines/wan_video.py``) and ``inject_lora`` (``fake_score_wan.py``).

The attention pattern is deliberately pluggable: :data:`SPARSE_PATTERNS` maps a
name to a ``mask_mod`` factory, so a new pattern is a one-function registration
and needs no change to the model, the trainer, or the checkpoint format.

This module intentionally does not import ``diffsynth`` at module scope, so the
CPU test suite can exercise it on a machine without the Wan dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    FLEX_ATTENTION_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the torch build
    create_block_mask = None
    flex_attention = None
    FLEX_ATTENTION_AVAILABLE = False


# Eager flex_attention materializes the full score matrix, which at Wan's token
# count is both ruinously slow and larger than the memory the sparsity was meant
# to save.  Compiling is what turns the block mask into actual skipped work.
# Set CACHEHEAD_COMPILE_FLEX=0 to fall back to eager while debugging (compiled
# flex_attention interacts with activation checkpointing and DDP, so having an
# escape hatch matters).
_COMPILED_FLEX_ATTENTION = None


def compiled_flex_attention():
    """Compile flex_attention once, lazily, and reuse it everywhere."""
    global _COMPILED_FLEX_ATTENTION
    if os.environ.get("CACHEHEAD_COMPILE_FLEX", "1") == "0":
        return flex_attention
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=False)
    return _COMPILED_FLEX_ATTENTION


# A materialized [S, S] boolean mask costs S^2 bytes.  At Wan's production token
# count (S = 32760) that is 1.07 GiB, so the dense-mask fallback is restricted to
# the small grids used by tests rather than silently exhausting memory.
MAX_DENSE_MASK_TOKENS = 8192

# flex_attention's block granularity.  Sparsity is only realized at block
# resolution, so a window narrower than one block still pays for a full block.
DEFAULT_BLOCK_SIZE = 128


# ═══════════════════════════════════════════════════════════════
# Pattern specification
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SparsePatternSpec:
    """Which sparse pattern to apply, and its shape.

    ``spatial_radius`` and ``temporal_radius`` are Chebyshev radii on the token
    grid, so ``spatial_radius=2`` is the 5x5 within-frame window.
    """

    name: str = "dense"
    spatial_radius: int = 2
    temporal_radius: int = 2

    def __post_init__(self) -> None:
        if self.name not in SPARSE_PATTERNS:
            raise ValueError(
                f"unknown sparse pattern {self.name!r}; available: {sorted(SPARSE_PATTERNS)}"
            )
        if self.spatial_radius < 0:
            raise ValueError(f"spatial_radius must be >= 0, got {self.spatial_radius}")
        if self.temporal_radius < 0:
            raise ValueError(f"temporal_radius must be >= 0, got {self.temporal_radius}")

    @property
    def is_dense(self) -> bool:
        return self.name == "dense"

    def window_size(self) -> int:
        """Keys visible to an interior query, before grid-edge clipping."""
        return (2 * self.spatial_radius + 1) ** 2 * (2 * self.temporal_radius + 1)


# ═══════════════════════════════════════════════════════════════
# Pattern registry
# ═══════════════════════════════════════════════════════════════
#
# A pattern factory takes (spec, grid) and returns a ``mask_mod`` compatible with
# torch's flex_attention: a callable ``(b, h, q_idx, kv_idx) -> bool`` broadcast
# over index tensors.  The patterns here are position-only, so ``b`` and ``h`` are
# ignored; they stay in the signature to satisfy flex_attention's contract.
#
# Token layout is Wan's: frame-major, then row-major, then column-major, i.e.
# ``index = frame * (height * width) + row * width + col``.  This matches the
# rope frequency construction in ``WanModel.forward``.

MaskMod = Callable[..., torch.Tensor]


def _decode(index: torch.Tensor, grid: tuple[int, int, int]):
    """Flat token index -> (frame, row, col) on Wan's token grid."""
    _, height, width = grid
    plane = height * width
    return index // plane, (index % plane) // width, index % width


def dense_mask_mod(spec: SparsePatternSpec, grid: tuple[int, int, int]) -> MaskMod:
    """Every query sees every key.  Present so ``dense`` is a valid registry
    entry; :class:`SparseAttentionModule` short-circuits it to the teacher kernel
    instead of paying for a mask."""

    def mask_mod(b, h, q_idx, kv_idx):
        # Touch both index tensors so the result broadcasts to [Q, KV]; returning
        # a q-only expression would silently yield a [Q, 1] mask.
        return (q_idx >= 0) & (kv_idx >= 0)

    return mask_mod


def spatiotemporal_window_mask_mod(spec: SparsePatternSpec, grid: tuple[int, int, int]) -> MaskMod:
    """A query at (f, r, c) attends to keys inside a Chebyshev box around it:
    ``|dr| <= spatial_radius``, ``|dc| <= spatial_radius``, ``|df| <= temporal_radius``."""
    spatial_radius = spec.spatial_radius
    temporal_radius = spec.temporal_radius

    def mask_mod(b, h, q_idx, kv_idx):
        q_frame, q_row, q_col = _decode(q_idx, grid)
        k_frame, k_row, k_col = _decode(kv_idx, grid)
        return (
            ((q_frame - k_frame).abs() <= temporal_radius)
            & ((q_row - k_row).abs() <= spatial_radius)
            & ((q_col - k_col).abs() <= spatial_radius)
        )

    return mask_mod


SPARSE_PATTERNS: dict[str, Callable[[SparsePatternSpec, tuple[int, int, int]], MaskMod]] = {
    "dense": dense_mask_mod,
    "spatiotemporal_window": spatiotemporal_window_mask_mod,
}


def build_mask_mod(spec: SparsePatternSpec, grid: tuple[int, int, int]) -> MaskMod:
    return SPARSE_PATTERNS[spec.name](spec, grid)


def materialize_mask(spec: SparsePatternSpec, grid: tuple[int, int, int], device=None) -> torch.Tensor:
    """Build the explicit ``[S, S]`` boolean mask.  This is the reference
    implementation the tests compare against, and the fallback used at small
    ``S`` when flex_attention is unavailable."""
    num_tokens = grid[0] * grid[1] * grid[2]
    if num_tokens > MAX_DENSE_MASK_TOKENS:
        raise ValueError(
            f"refusing to materialize a {num_tokens}x{num_tokens} mask "
            f"({num_tokens ** 2 / 2**30:.2f} GiB); flex_attention is required at this token count"
        )
    mask_mod = build_mask_mod(spec, grid)
    index = torch.arange(num_tokens, device=device)
    return mask_mod(None, None, index.unsqueeze(-1), index.unsqueeze(0))


# ═══════════════════════════════════════════════════════════════
# Attention kernels
# ═══════════════════════════════════════════════════════════════

def _teacher_dense_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Exactly the kernel the teacher uses, so ``dense`` is comparable to it.

    Wan picks a backend at import time (flash-attn 3/2, sage, or SDPA) and those
    differ numerically, so parity requires calling Wan's own dispatcher when it
    is importable rather than reimplementing SDPA here.
    """
    try:
        from diffsynth.models.wan_video_dit import flash_attention
    except Exception:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        out = F.scaled_dot_product_attention(q, k, v)
        return rearrange(out, "b n s d -> b s (n d)", n=num_heads)
    return flash_attention(q=q, k=k, v=v, num_heads=num_heads)


class SparseAttentionModule(nn.Module):
    """Drop-in replacement for ``wan_video_dit.AttentionModule``.

    Contract, unchanged from the module it replaces: ``forward(q, k, v)`` with
    each of shape ``[B, S, dim]`` (heads still packed into the last axis),
    returning ``[B, S, dim]``.  Parameter-free, so installing it does not alter
    the inherited ``state_dict``.

    ``forward`` receives no positional information, so the owning model stamps
    the token grid on every instance before running the DiT (see
    :func:`set_token_grid`).  This mirrors how the RAS code stashes
    ``_last_selected_patches`` on the model between calls.
    """

    def __init__(self, num_heads: int, spec: SparsePatternSpec, block_size: int = DEFAULT_BLOCK_SIZE):
        super().__init__()
        self.num_heads = num_heads
        self.spec = spec
        self.block_size = block_size
        self.grid: tuple[int, int, int] | None = None
        # The mask depends only on (pattern, grid, device): it is identical
        # across every block and every denoising step.  Rebuilding it per call
        # would dominate the runtime the sparsity is meant to save.
        self._mask_cache: dict = {}

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, pattern={self.spec.name}"

    def set_grid(self, grid: tuple[int, int, int] | None) -> None:
        self.grid = None if grid is None else tuple(int(d) for d in grid)

    def _require_grid(self, num_tokens: int) -> tuple[int, int, int]:
        if self.grid is None:
            raise RuntimeError(
                "SparseAttentionModule.grid is unset; call set_token_grid(...) on the "
                "owning model before the DiT forward"
            )
        expected = self.grid[0] * self.grid[1] * self.grid[2]
        if expected != num_tokens:
            raise ValueError(
                f"token grid {self.grid} implies {expected} tokens but attention received {num_tokens}"
            )
        return self.grid

    def _block_mask(self, grid: tuple[int, int, int], num_tokens: int, device: torch.device):
        key = ("block", grid, num_tokens, device.type, device.index, self.block_size)
        cached = self._mask_cache.get(key)
        if cached is None:
            cached = create_block_mask(
                build_mask_mod(self.spec, grid),
                B=None,
                H=None,
                Q_LEN=num_tokens,
                KV_LEN=num_tokens,
                device=device,
                BLOCK_SIZE=self.block_size,
            )
            self._mask_cache[key] = cached
        return cached

    def _bool_mask(self, grid: tuple[int, int, int], num_tokens: int, device: torch.device):
        key = ("bool", grid, num_tokens, device.type, device.index)
        cached = self._mask_cache.get(key)
        if cached is None:
            cached = materialize_mask(self.spec, grid, device=device)
            self._mask_cache[key] = cached
        return cached

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.spec.is_dense:
            return _teacher_dense_attention(q, k, v, self.num_heads)

        if q.shape[1] != k.shape[1]:
            raise ValueError(
                f"sparse attention needs matching query/key lengths, got {q.shape[1]} and "
                f"{k.shape[1]}; the RAS token-gather path is not supported by the sparse student"
            )

        num_tokens = q.shape[1]
        grid = self._require_grid(num_tokens)

        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)

        if FLEX_ATTENTION_AVAILABLE and num_tokens >= self.block_size:
            out = compiled_flex_attention()(
                q, k, v, block_mask=self._block_mask(grid, num_tokens, q.device)
            )
        else:
            # Small grids (the CPU tests) or a torch build without flex_attention.
            mask = self._bool_mask(grid, num_tokens, q.device)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        return rearrange(out, "b n s d -> b s (n d)", n=self.num_heads)


def install_sparse_attention(
    dit: nn.Module, spec: SparsePatternSpec, block_size: int = DEFAULT_BLOCK_SIZE
) -> int:
    """Replace every self-attention kernel in ``dit`` with a sparse one, in place.

    Cross-attention is deliberately left alone: it attends video tokens to text
    tokens, where the spatial grid has no meaning.  Returns the number of blocks
    converted.
    """
    count = 0
    for block in dit.blocks:
        block.self_attn.attn = SparseAttentionModule(
            block.self_attn.num_heads, spec, block_size=block_size
        )
        count += 1
    return count


def set_token_grid(dit: nn.Module, grid: tuple[int, int, int] | None) -> None:
    """Stamp the current token grid on every sparse attention module in ``dit``."""
    for block in dit.blocks:
        attn = block.self_attn.attn
        if isinstance(attn, SparseAttentionModule):
            attn.set_grid(grid)
