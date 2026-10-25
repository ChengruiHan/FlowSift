"""Atomic complete-group checkpoints; never reconstruct reference from current."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from flowvla_rl.training import tree_to


@dataclass
class RunState:
    attempted_groups: int = 0
    episodes: int = 0
    env_steps: int = 0
    completed_updates: int = 0
    collapsed_groups: int = 0
    elapsed_seconds: float = 0.0


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA checkpoint requires CUDA for exact RNG restoration")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        t = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(t.shape), t.dtype)).encode())
        digest.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(root, policy, optimizer, state, metadata, *, final=False, export=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=root))
    name = f"{'final' if final else 'attempt'}_{state.attempted_groups:06d}_{uuid.uuid4().hex[:8]}"
    destination = root / name
    try:
        payload = {"schema_version": 1, "current": tree_to(policy.state_dict(), "cpu"),
                   "optimizer": tree_to(optimizer.state_dict(), "cpu"),
                   "run_state": asdict(state), "metadata": metadata, "rng": capture_rng()}
        torch.save(payload, temporary / "training_state.pt")
        if export:
            export(temporary / "pretrained_model")
        atomic_json(temporary / "metadata.json", metadata)
        os.replace(temporary, destination)
        atomic_json(root / "latest.json", {"path": name})
        if final:
            atomic_json(root / "final.json", {"path": name})
        # Only rotate checkpoints owned by this function, after latest is committed.
        for old in root.glob("attempt_*"):
            if old != destination and old.is_dir() and (old / "metadata.json").exists():
                shutil.rmtree(old)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_checkpoint(path):
    path = Path(path)
    if path.suffix == ".json":
        path = path.parent / json.loads(path.read_text())["path"]
    if path.is_dir():
        path = path / "training_state.pt"
    # Only load trusted local artifacts: torch checkpoints include Python/NumPy RNG objects.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["schema_version"] != 1:
        raise ValueError("unsupported checkpoint schema")
    return saved


def restore_checkpoint(saved, policy, optimizer, expected_metadata, *, restore_random=True):
    if saved["metadata"] != expected_metadata:
        raise ValueError("checkpoint protocol, model/reference, code or RNG metadata mismatch")
    policy.load_state_dict(saved["current"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    if restore_random:
        restore_rng(saved["rng"])
    return RunState(**saved["run_state"])
