"""Small, torchrun-compatible DDP helpers for CacheHead training.

Launch the training harness with ``torchrun --standalone --nproc_per_node=8``.
When ``WORLD_SIZE`` is one (the normal ``python`` invocation), every helper is
intentionally a no-op and the harness remains single-GPU compatible.
"""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    """Process metadata plus collectives used by the training harness."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def wrap(self, module: torch.nn.Module) -> torch.nn.Module:
        """Wrap a same-device module so its trainable gradients are averaged."""
        if not self.enabled:
            return module
        return DistributedDataParallel(
            module,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    @staticmethod
    def unwrap(module: torch.nn.Module) -> torch.nn.Module:
        return module.module if isinstance(module, DistributedDataParallel) else module

    @contextmanager
    def no_sync(self, module: torch.nn.Module, enabled: bool) -> Iterator[None]:
        """Defer DDP gradient all-reduction during intermediate accumulation."""
        context = module.no_sync() if self.enabled and enabled else nullcontext()
        with context:
            yield

    def all_true(self, value: bool) -> bool:
        """Return true only when every rank supplied true."""
        if not self.enabled:
            return value
        flag = torch.tensor(int(value), device=self.device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def max_int(self, value: int) -> int:
        """Synchronize a loop count to the maximum requested by any rank."""
        if not self.enabled:
            return value
        result = torch.tensor(value, device=self.device, dtype=torch.int64)
        dist.all_reduce(result, op=dist.ReduceOp.MAX)
        return int(result.item())

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def cleanup(self) -> None:
        if self.enabled and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def initialize_distributed(requested_device: str | None = None) -> DistributedContext:
    """Initialize NCCL from torchrun's environment and select the local A100.

    ``torchrun`` defines ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``.  A DDP
    process always uses its ``LOCAL_RANK`` device; accepting a fixed
    ``cuda:N`` here would silently put multiple ranks on one GPU.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        if requested_device is not None:
            requested = torch.device(requested_device)
            if requested.type != "cuda" or requested.index is not None:
                raise ValueError(
                    "DDP selects cuda:LOCAL_RANK. Use --device cuda or omit --device; "
                    "do not pin a GPU index."
                )
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested by WORLD_SIZE > 1, but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        return DistributedContext(rank, local_rank, world_size, torch.device(f"cuda:{local_rank}"))

    device_name = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    return DistributedContext(rank, local_rank, world_size, torch.device(device_name))
