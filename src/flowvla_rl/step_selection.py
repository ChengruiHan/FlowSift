"""FlowSift v3: reproducible, per-chunk selection independent of rollout RNG."""
from __future__ import annotations

import hashlib
import json
from numbers import Integral
from typing import Iterable

import torch

RNG_VERSION = "sha256_v1"
METHODS = ("full10", "late5", "uniform5")


def selection_seed(train_seed: int) -> int:
    digest = hashlib.sha256(f"FlowSift-v3-selection:{train_seed}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63)


def validate_indices(indices: Iterable[int] | None, steps: int) -> tuple[int, ...]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if indices is None:
        return tuple(range(steps))
    values = tuple(indices)
    if not values or any(isinstance(k, bool) or not isinstance(k, Integral) for k in values):
        raise ValueError("step_indices must contain nonempty integer indices")
    if len(set(values)) != len(values) or any(k < 0 or k >= steps for k in values):
        raise ValueError("step_indices must be unique and in [0, K)")
    return tuple(int(k) for k in values)


def select_steps(method: str, *, root_seed: int, attempted_group_id: int,
                 episode_id: int, chunk_id: int, steps: int = 10) -> tuple[int, ...]:
    if method not in METHODS or steps != 10:
        raise ValueError("v3 requires full10/late5/uniform5 on the original K=10 grid")
    if min(root_seed, attempted_group_id, episode_id, chunk_id) < 0:
        raise ValueError("selection seeds and stable IDs must be nonnegative")
    if method == "full10":
        return tuple(range(10))
    if method == "late5":
        return tuple(range(5, 10))
    key = json.dumps([RNG_VERSION, root_seed, attempted_group_id, episode_id, chunk_id],
                     separators=(",", ":")).encode()
    local_seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    return tuple(sorted(torch.randperm(10, generator=generator)[:5].tolist()))
