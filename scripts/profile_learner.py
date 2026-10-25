#!/usr/bin/env python
"""Paired fixed-cache learner measurements; isolated processes for GPU peaks."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from contextlib import contextmanager

import torch

from flowvla_rl.checkpointing import atomic_json, state_hash
from flowvla_rl.diagnostics import gradient_probe
from flowvla_rl.reference import make_frozen_reference
from flowvla_rl.step_selection import METHODS, selection_seed
from flowvla_rl.training import optimize_grpo, synchronize


@contextmanager
def count_denoise_calls(model):
    # Do not retain call arguments/caches as Mock would: that corrupts peak memory.
    original = model.denoise_step
    counter = [0]
    def counted(*args, **kwargs):
        counter[0] += 1
        return original(*args, **kwargs)
    model.denoise_step = counted
    try:
        yield counter
    finally:
        model.denoise_step = original


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--caches", type=Path)
    p.add_argument("--sft")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, choices=[3, 5], default=3)
    p.add_argument("--toy", action="store_true", help="CPU integration demo; not SmolVLA results")
    p.add_argument("--method", choices=METHODS, help=argparse.SUPPRESS)
    p.add_argument("--round", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--gradient-probe", action="store_true")
    return p.parse_args()


def worker(args):
    runtime = None
    if args.toy:
        from flowvla_rl.toy_backend import TinyPolicy, make_chunks
        torch.manual_seed(7)
        policy = TinyPolicy()
        reference = make_frozen_reference(policy.model)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-6, weight_decay=1e-10)
        optimize_grpo(policy, reference, make_chunks(policy), optimizer, clip_epsilon=0.2,
                      kl_beta=0.01, epochs=1, max_grad_norm=1, seed=7)
        bundles = [{"chunks": make_chunks(policy, 20 + i), "current": copy.deepcopy(policy.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()), "attempted_group_id": i,
                    "train_seed": 7, "task_id": i, "policy_hash": state_hash(policy)} for i in range(3)]
    else:
        if not args.caches or not args.sft or not torch.cuda.is_available():
            raise ValueError("real profiling requires --caches, --sft and CUDA")
        from flowvla_rl.libero_rollout import make_runtime
        runtime = make_runtime(args.sft, "libero_object", task_id=0)
        policy = runtime.policy
        reference = make_frozen_reference(policy.model)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-6, weight_decay=1e-10)
        bundles = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(args.caches.glob("cache_*.pt"))]
        if len(bundles) != 3 or len({b["task_id"] for b in bundles}) != 3:
            raise ValueError("profiling requires exactly three mixed caches from three tasks")
        ref_hash = state_hash(reference)
        if any(b["metadata"]["reference_hash"] != ref_hash for b in bundles):
            raise ValueError("cache reference hash does not match pinned SFT")
    device = next(policy.parameters()).device
    def restore(bundle):
        policy.load_state_dict(bundle["current"])
        optimizer.load_state_dict(copy.deepcopy(bundle["optimizer"]))
    def update(bundle, chunks=None, detailed=False):
        return optimize_grpo(policy, reference, bundle["chunks"] if chunks is None else chunks, optimizer,
                             clip_epsilon=0.2, kl_beta=0.01, epochs=1, max_grad_norm=1,
                             seed=bundle["train_seed"] + bundle["attempted_group_id"], method=args.method,
                             root_seed=selection_seed(bundle["train_seed"]),
                             attempted_group_id=bundle["attempted_group_id"], detailed_timing=detailed)
    rows, detail_rows = [], []
    try:
        # Warmup update for this mode, then restore each exact pre-update snapshot.
        restore(bundles[0])
        if not optimizer.state:
            raise ValueError("steady-state profiling requires initialized Adam state")
        update(bundles[0])
        synchronize(device)
        initialization_peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        for i, bundle in enumerate(bundles):
            restore(bundle)
            if state_hash(policy) != bundle["policy_hash"]:
                raise ValueError("profile current snapshot hash mismatch")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            with count_denoise_calls(policy.model) as current_calls, count_denoise_calls(reference) as ref_calls:
                metrics = update(bundle)
                actual_current, actual_ref = current_calls[0], ref_calls[0]
            expected = len(bundle["chunks"]) * (10 if args.method == "full10" else 5)
            if (actual_current, actual_ref) != (expected, expected):
                raise AssertionError("actual denoise calls differ from selected count")
            rows.append({"backend": "toy_cpu" if args.toy else "smolvla_cuda", "method": args.method,
                         "cache_id": i, "task_id": bundle["task_id"], "repeat": args.round,
                         "learner_ms": metrics["learner_seconds"] * 1000,
                         "chunk_count": len(bundle["chunks"]), "current_calls": actual_current,
                         "reference_calls": actual_ref,
                         "allocated_peak": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
                         "reserved_peak": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
                         "initialization_allocated_peak": initialization_peak,
                         "trajectory_bytes": sum(c.trajectory.states.numel() * c.trajectory.states.element_size()
                            + c.trajectory.element_log_probs.numel() * c.trajectory.element_log_probs.element_size()
                            + c.trajectory.valid_action_mask.numel() * c.trajectory.valid_action_mask.element_size()
                            for c in bundle["chunks"])})
        # Separate pass: exactly 16 fixed chunks across available snapshots on real caches.
        if args.round == 0:
            remaining = 16
            for i, bundle in enumerate(bundles):
                chosen = bundle["chunks"][:remaining]
                if not chosen:
                    break
                restore(bundle)
                details = update(bundle, chunks=chosen, detailed=True)
                detail_rows.append({"method": args.method, "cache_id": i, "chunks": len(chosen),
                                    **{k: v for k, v in details.items() if k.endswith("_seconds")}})
                remaining -= len(chosen)
            if remaining > 0 and not args.toy:
                raise ValueError("real caches must provide at least 16 chunks for detail pass")
        if args.gradient_probe and args.method == "full10" and args.round == 0:
            probe_rows = []
            for i, bundle in enumerate(bundles):
                restore(bundle)
                result = gradient_probe(policy, reference, bundle["chunks"][:1], selection_seed(7))
                probe_rows.extend({"cache_id": i, **r} for r in result)
            atomic_json(args.output / "gradient_probe.json", probe_rows)
        atomic_json(args.output / f"worker_{args.method}_{args.round}.json", {"rows": rows, "details": detail_rows})
    finally:
        if runtime is not None:
            runtime.close()


def main():
    args = arguments()
    if args.method:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    for repeat in range(args.repeats):
        order = METHODS[repeat % 3:] + METHODS[:repeat % 3]
        for method in order:
            command = [sys.executable, __file__, "--output", str(args.output), "--method", method,
                       "--round", str(repeat)]
            if args.toy:
                command.append("--toy")
            else:
                if not args.caches or not args.sft:
                    raise ValueError("provide --caches and --sft")
                command += ["--caches", str(args.caches), "--sft", args.sft]
            if args.gradient_probe:
                command.append("--gradient-probe")
            subprocess.run(command, check=True)
    all_rows, details = [], []
    for path in sorted(args.output.glob("worker_*.json")):
        result = json.loads(path.read_text())
        all_rows.extend(result["rows"])
        details.extend(result["details"])
    for filename, rows in (("learner_profile.csv", all_rows), ("learner_phases.csv", details)):
        with (args.output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Saved {len(all_rows)} measurements to {args.output}; backend={'toy_cpu' if args.toy else 'smolvla_cuda'}")


if __name__ == "__main__":
    main()
