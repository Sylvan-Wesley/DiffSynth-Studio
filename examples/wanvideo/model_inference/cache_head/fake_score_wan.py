"""
Fake-score estimator for Strict DMD training: a *LoRA Wan*.

Mirrors DMD2's ``mu_fake <- copyWeights(mu_real)`` (the fake-score estimator is
a copy of the teacher diffusion model) but parameterizes the copy cheaply as
frozen Wan DiT + a trainable low-rank adapter:

    v_fake(x_t, t, ctx) = Wan_LoRA(x_t, t, ctx)

The base clone is frozen; only the LoRA ``lora_A``/``lora_B`` matrices train.
The LoRA delta is zero at initialization, so the fake-score starts identical to
the teacher and only diverges as the adapter learns the hybrid generator's
score from stop-gradient generated samples.

This module is a training-only auxiliary.  It is discarded after training and
never appears in the exported CacheHead checkpoint.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """nn.Linear augmented with a low-rank adapter: ``y = Wx + alpha*(B@A@x)``.

    ``base`` is kept frozen; only ``lora_A``/``lora_B`` hold trainable weights.
    ``lora_B`` is zero-initialized so the adapter delta is exactly zero at init.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps nn.Linear, got {type(base).__name__}")
        self.base = base
        in_features, out_features = base.in_features, base.out_features
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B stays zero -> delta == 0 at init -> fake_score == teacher.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + self.scaling * delta.to(x.dtype)


def inject_lora(module: nn.Module, rank: int, alpha: float) -> None:
    """Replace every ``nn.Linear`` in ``module`` in place with a ``LoRALinear``."""
    for name, child in list(module._modules.items()):
        if isinstance(child, nn.Linear):
            module._modules[name] = LoRALinear(child, rank, alpha)
        else:
            inject_lora(child, rank, alpha)


class FakeScoreWan(nn.Module):
    """LoRA Wan: a deep-copied, frozen Wan DiT with trainable low-rank adapters.

    Forward is the ordinary WanModel forward (same signature), so the harness
    drives positive/negative CFG exactly as it does for the teacher.  Only LoRA
    A/B parameters require gradients.
    """

    def __init__(self, dit: nn.Module, rank: int = 32, alpha: float = 1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        # Separate, frozen clone so the teacher DiT (and any shared buffers)
        # is never mutated.
        base = copy.deepcopy(dit)
        base.eval()
        inject_lora(base, rank, alpha)
        self.base = base
        self._freeze_base_keep_lora()

    def _freeze_base_keep_lora(self) -> None:
        for name, param in self.named_parameters():
            is_lora = ".lora_A" in name or ".lora_B" in name
            param.requires_grad_(is_lora)

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def lora_parameters(self):
        for name, param in self.named_parameters():
            if ".lora_A" in name or ".lora_B" in name:
                yield param
