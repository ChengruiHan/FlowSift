#!/usr/bin/env python
"""Prepare LIBERO's configuration without triggering its interactive first import."""
import importlib.metadata
import os
from pathlib import Path
import yaml

if __name__ == "__main__":
    distribution = importlib.metadata.distribution("hf-libero")
    candidates = [Path(distribution.locate_file(p)).parent for p in distribution.files or []
                  if str(p).endswith("libero/libero/__init__.py")]
    if len(candidates) != 1:
        raise RuntimeError("cannot locate hf-libero benchmark root; inspect installed package")
    root = candidates[0]
    destination = Path(os.environ.get("LIBERO_CONFIG_PATH", str(Path.home() / ".libero"))) / "config.yaml"
    if destination.exists():
        print(f"Using existing LIBERO configuration: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        datasets = Path.cwd() / "data" / "libero"
        datasets.mkdir(parents=True, exist_ok=True)
        destination.write_text(yaml.safe_dump({"benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
            "init_states": str(root / "init_files"), "datasets": str(datasets), "assets": str(root / "assets")}))
        print(destination)
