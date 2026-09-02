"""SparseCacheHead: a teacher-inherited Wan DiT with sparse self-attention.

Where the original ``CacheHead`` is a ~72k-parameter token-space residual net,
this variant *is* the teacher, structurally:

    student = deepcopy(teacher DiT)
              -> optionally keep a subset of blocks (shallower)
              -> every self-attention swapped for a sparse one
              -> a conv3d that fuses the previous guided velocity into the input

The hypothesis is that a head step does not need dense global attention, because
the previous guided prediction already carries the long-range structure the
previous *dense* step computed.  The conv3d is how that information reaches the
DiT, and sparse attention is what makes the step cheap.

Input contract, matching ``debug-training``'s latent-conditioned variants:

    prev_guided_tokens  [B, S, 64]        CFG-guided velocity from the previous step
    current_latents     [B, 16, f, h, w]  the latent x_k at the start of this step

Both are mapped into the 16-channel latent space, concatenated, compressed back
to 16 channels by a small conv3d, and added residually to ``current_latents``.
The conv's last layer is zero-initialized, so at step 0 the fused input is
*exactly* ``current_latents`` and the DiT sees precisely the distribution it was
trained on.  That is why no warm-up phase is needed.

The model predicts a **residual** on the previous guided velocity, matching
``CacheHead``'s convention::

    v_hat_k = v_{k-1} + student(v_{k-1}, x_k, t_k, ctx_posi)

The DiT's output head is zero-initialized too, so a fresh student emits exactly
zero and reproduces ``carry_previous`` bit-for-bit.  That zeroing is not
optional for a residual formulation: keeping the teacher's head would yield
``v_{k-1} + v_posi(x_k)``, roughly double the correct magnitude.

A head step is a single forward with positive context only, with classifier-free
guidance distilled into the weights.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn as nn

from cache_head_model import (
    SPARSE_DIT_VARIANT,
    CacheHeadConfig,
    _deep_asdict,
    read_cache_head_checkpoint,
    unpatchify_tokens,
)
from sparse_attention import (
    DEFAULT_BLOCK_SIZE,
    SparsePatternSpec,
    install_sparse_attention,
    set_token_grid,
)


# ═══════════════════════════════════════════════════════════════
# Depth selection
# ═══════════════════════════════════════════════════════════════

def resolve_layer_indices(
    num_teacher_layers: int,
    num_layers: int | None = None,
    explicit=None,
) -> tuple[int, ...]:
    """Pick which teacher blocks the student keeps.

    Dropping blocks is well-defined because each ``DiTBlock`` is a pure residual
    update on a dimensionally-invariant stream, so a subset composes exactly like
    the full stack, just with less capacity.

    ``explicit`` wins when given.  Otherwise ``num_layers`` selects a uniform
    stride that always retains the first and last block -- those carry the most
    in a DiT, and the endpoints anchor the residual stream.  Prefer choosing the
    indices from measurement (``profile_block_importance.py``) over the stride.
    """
    if explicit is not None:
        indices = tuple(int(i) for i in explicit)
        if len(indices) == 0:
            raise ValueError("explicit layer indices must be non-empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"explicit layer indices must be unique, got {indices}")
        if list(indices) != sorted(indices):
            raise ValueError(f"explicit layer indices must be sorted ascending, got {indices}")
        if indices[0] < 0 or indices[-1] >= num_teacher_layers:
            raise ValueError(
                f"explicit layer indices {indices} out of range for a "
                f"{num_teacher_layers}-block teacher"
            )
        return indices

    if num_layers is None or num_layers == num_teacher_layers:
        return tuple(range(num_teacher_layers))
    if not 1 <= num_layers <= num_teacher_layers:
        raise ValueError(
            f"num_layers must be in [1, {num_teacher_layers}], got {num_layers}"
        )
    if num_layers == 1:
        return (0,)
    step = (num_teacher_layers - 1) / (num_layers - 1)
    indices = tuple(round(i * step) for i in range(num_layers))
    # step >= 1 whenever num_layers <= num_teacher_layers, so the rounded
    # positions are strictly increasing; assert rather than trust the arithmetic.
    if len(set(indices)) != num_layers:
        raise RuntimeError(f"stride selection produced duplicates: {indices}")
    return indices


# ═══════════════════════════════════════════════════════════════
# Input adapter
# ═══════════════════════════════════════════════════════════════

class LatentFusionConv3d(nn.Module):
    """Compress ``[prev_guided_latent ‖ current_latents]`` back to latent width.

    Output width equals ``dit.in_dim`` (16 for Wan2.1), so the fused tensor feeds
    the teacher's ``patch_embedding`` unchanged and that inherited layer keeps
    working on its native input.

    The final conv is zero-initialized, making the module an exact identity on
    ``current_latents`` at step 0.
    """

    def __init__(self, latent_channels: int = 16, hidden_channels: int = 64, kernel_size: int = 3):
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd int, got {kernel_size}")
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(2 * latent_channels, hidden_channels, kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, latent_channels, kernel_size, padding=padding),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, prev_guided_latent: torch.Tensor, current_latents: torch.Tensor) -> torch.Tensor:
        if prev_guided_latent.shape != current_latents.shape:
            raise ValueError(
                f"fusion inputs must match: prev_guided_latent {tuple(prev_guided_latent.shape)} "
                f"vs current_latents {tuple(current_latents.shape)}"
            )
        fused = torch.cat((prev_guided_latent, current_latents), dim=1)
        return current_latents + self.net(fused)


# ═══════════════════════════════════════════════════════════════
# The student
# ═══════════════════════════════════════════════════════════════

class SparseCacheHead(nn.Module):
    """A sparse-attention Wan DiT that predicts the guided velocity at a head step.

    ``teacher_dit`` is deep-copied (never mutated), so the student starts from the
    teacher's weights exactly -- the same cloning recipe ``FakeScoreWan`` uses.
    """

    def __init__(
        self,
        teacher_dit: nn.Module,
        config: CacheHeadConfig,
        *,
        use_gradient_checkpointing: bool = True,
        attention_block_size: int = DEFAULT_BLOCK_SIZE,
        zero_init_output: bool = True,
    ):
        super().__init__()
        if config.head_variant != SPARSE_DIT_VARIANT:
            raise ValueError(
                f"SparseCacheHead requires head_variant '{SPARSE_DIT_VARIANT}', "
                f"got {config.head_variant!r}"
            )
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Clone first, then mutate the clone.  The teacher keeps its full depth,
        # its dense attention, and its frozen parameters.
        dit = copy.deepcopy(teacher_dit)
        # The caller freezes the teacher (``dit.requires_grad_(False)``) before we
        # get here, and deepcopy carries ``requires_grad`` across -- so without
        # this the student would silently train nothing at all.
        dit.requires_grad_(True)

        num_teacher_layers = len(dit.blocks)
        indices = resolve_layer_indices(
            num_teacher_layers, explicit=config.student_layer_indices
        )
        if tuple(indices) != tuple(range(num_teacher_layers)):
            dit.blocks = nn.ModuleList([dit.blocks[i] for i in indices])
        self.layer_indices = tuple(indices)
        self.num_teacher_layers = num_teacher_layers

        self.pattern = SparsePatternSpec(
            name=config.sparse_pattern,
            spatial_radius=config.sparse_spatial_radius,
            temporal_radius=config.sparse_temporal_radius,
        )
        install_sparse_attention(dit, self.pattern, block_size=attention_block_size)

        # The student predicts a residual on the previous guided velocity, so a
        # fresh student must emit exactly zero.  Keeping the teacher's output
        # head instead would give ``prev_guided + v_posi(x_k)`` -- roughly double
        # the correct magnitude -- so zeroing it is what makes the residual
        # formulation start from the carry_previous baseline.
        self.zero_init_output = zero_init_output
        if zero_init_output:
            nn.init.zeros_(dit.head.head.weight)
            nn.init.zeros_(dit.head.head.bias)

        self.dit = dit
        self.patch_size = tuple(int(p) for p in teacher_dit.patch_size)
        self.fusion = LatentFusionConv3d(
            latent_channels=config.latent_channels,
            hidden_channels=config.fusion_hidden_channels,
            kernel_size=config.fusion_kernel_size,
        )

    # -- plumbing ------------------------------------------------------

    def set_token_grid(self, grid) -> None:
        """Stamp the token grid on every sparse attention module.

        ``AttentionModule.forward(q, k, v)`` carries no positional information,
        so the grid has to be pushed down out of band before the DiT runs.
        """
        set_token_grid(self.dit, grid)

    def fuse(self, prev_guided_tokens: torch.Tensor, current_latents: torch.Tensor, grid) -> torch.Tensor:
        """``[B,S,64]`` + ``[B,16,f,h,w]`` -> the ``[B,16,f,h,w]`` DiT input."""
        prev_latent = unpatchify_tokens(prev_guided_tokens, grid, self.patch_size)
        return self.fusion(prev_latent, current_latents)

    # -- forward -------------------------------------------------------

    def forward(
        self,
        prev_guided_tokens: torch.Tensor,
        current_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        grid,
    ) -> torch.Tensor:
        """Return the predicted *residual* ``[B, S, 64]``.

        The caller adds it to ``prev_guided_tokens``, matching ``CacheHead``'s
        convention:  ``v_hat = prev_guided + student(...)``.  With
        ``zero_init_output`` the residual is exactly zero at init, so a fresh
        student reproduces ``carry_previous`` bit-for-bit.

        ``context`` is the *positive* prompt embedding only: CFG is distilled
        into the weights, so a head step is a single forward.
        """
        fused = self.fuse(prev_guided_tokens, current_latents, grid)
        self.set_token_grid(grid)
        _, noise_tokens = self.dit(
            x=fused,
            timestep=timestep,
            context=context,
            return_noise_tokens=True,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
        )
        return noise_tokens  # the residual; the caller adds prev_guided

    # -- reporting -----------------------------------------------------

    def parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        fusion = sum(p.numel() for p in self.fusion.parameters())
        return {
            "total_parameters": total,
            "fusion_parameters": fusion,
            "dit_parameters": total - fusion,
            "num_layers": len(self.dit.blocks),
            "num_teacher_layers": self.num_teacher_layers,
            "layer_indices": self.layer_indices,
            "sparse_pattern": self.pattern.name,
        }


# ═══════════════════════════════════════════════════════════════
# Checkpoint I/O
# ═══════════════════════════════════════════════════════════════

def save_sparse_cache_head(model: nn.Module, config: CacheHeadConfig, path: str | Path) -> Path:
    """Persist the student.

    The block subset is baked into ``config.student_layer_indices`` so loading
    rebuilds the identical architecture.  Note this writes ~1.3B parameters
    (~2.6 GB in bf16), unlike the ~300 KB legacy CacheHead checkpoints.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if config.head_variant != SPARSE_DIT_VARIANT:
        raise ValueError(
            f"save_sparse_cache_head expects head_variant '{SPARSE_DIT_VARIANT}', "
            f"got {config.head_variant!r}"
        )
    payload = {
        "version": 4,
        "config": _deep_asdict(config),
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    }
    torch.save(payload, path)
    return path


def load_sparse_cache_head(
    path: str | Path,
    teacher_dit: nn.Module,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
    *,
    use_gradient_checkpointing: bool = False,
) -> tuple[SparseCacheHead, CacheHeadConfig]:
    """Rebuild a :class:`SparseCacheHead` from a checkpoint.

    ``teacher_dit`` supplies the module *structure* only -- every parameter is
    then overwritten by the checkpoint -- but it must be the same Wan variant the
    student was cloned from, or the state dict will not match.
    """
    config, state_dict = read_cache_head_checkpoint(path)
    if config.head_variant != SPARSE_DIT_VARIANT:
        raise ValueError(
            f"not a sparse_dit checkpoint (head_variant={config.head_variant!r}); "
            "use cache_head_model.load_cache_head instead"
        )
    model = SparseCacheHead(
        teacher_dit, config, use_gradient_checkpointing=use_gradient_checkpointing
    )
    model.load_state_dict(state_dict)
    model.to(device=device) if dtype is None else model.to(device=device, dtype=dtype)
    model.eval()
    return model, config
