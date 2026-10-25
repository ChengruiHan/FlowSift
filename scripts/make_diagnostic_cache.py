#!/usr/bin/env python
"""Choose eight fixed non-test chunks from warmed Stage-A profile caches."""
import argparse
from pathlib import Path
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caches", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chunks = []
    for path in sorted(args.caches.glob("cache_*.pt")):
        chunks.extend(torch.load(path, map_location="cpu", weights_only=False)["chunks"])
    if len(chunks) < 8:
        raise ValueError("need at least eight Stage-A chunks")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        torch.save(chunks[:8], handle)
