"""Actual source, model, dependency and hardware provenance (no inferred hashes)."""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch

from flowvla_rl.checkpointing import file_hash
from flowvla_rl.protocol import canonical_hash


def git_commit(directory):
    result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                            capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def source_hash(root):
    files = [*Path(root).glob("src/**/*.py"), *Path(root).glob("scripts/*.py"), Path(root) / "pyproject.toml"]
    return canonical_hash({str(p.relative_to(root)): file_hash(p) for p in sorted(files)})


def model_files(directory):
    root = Path(directory)
    if not (root / "config.json").is_file() or not list(root.glob("*.safetensors")):
        raise ValueError("SFT directory must contain config.json and safetensors weights")
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
            if p.is_file() and ".cache" not in p.parts and p.name != "source_manifest.json"}


def environment_report(root):
    versions = {}
    for name in ("torch", "numpy", "lerobot", "mujoco", "robosuite", "hf-libero", "gymnasium", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {"python": platform.python_version(), "platform": platform.platform(),
              "dependency_versions": versions, "cuda": torch.version.cuda,
              "cpu_count": os.cpu_count(), "disk_free_bytes": shutil.disk_usage(root).free,
              "code_commit": git_commit(root), "source_hash": source_hash(root),
              "gpu": [], "driver": None, "ram_bytes": None, "cgroup_cpu_max": None,
              "cgroup_memory_max": None, "lerobot_commit": None}
    lerobot_spec = importlib.util.find_spec("lerobot")
    if lerobot_spec and lerobot_spec.submodule_search_locations:
        report["lerobot_commit"] = git_commit(next(iter(lerobot_spec.submodule_search_locations)))
    if hasattr(os, "sysconf"):
        try:
            report["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    for key, path in (("cgroup_cpu_max", "/sys/fs/cgroup/cpu.max"),
                      ("cgroup_memory_max", "/sys/fs/cgroup/memory.max")):
        if Path(path).is_file():
            report[key] = Path(path).read_text().strip()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            report["gpu"].append({"name": props.name, "total_memory": props.total_memory})
        if shutil.which("nvidia-smi"):
            result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                    capture_output=True, text=True)
            report["driver"] = result.stdout.strip()
    try:
        dist = importlib.metadata.distribution("lerobot")
        direct = dist.read_text("direct_url.json")
        report["lerobot_direct_url"] = direct
    except importlib.metadata.PackageNotFoundError:
        report["lerobot_direct_url"] = None
    return report
