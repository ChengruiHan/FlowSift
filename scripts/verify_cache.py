#!/usr/bin/env python
"""Server gate: compare fixed-cache scoring and Full updates against pinned upstream."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import torch

from flowvla_rl.checkpointing import atomic_json, state_hash
from flowvla_rl.provenance import source_hash
from flowvla_rl.reference import make_frozen_reference
from flowvla_rl.smolvla_sde import prepare_policy_prefix, recompute_log_probs, rollout_autocast
from flowvla_rl.step_selection import select_steps, selection_seed
from flowvla_rl.training import optimize_grpo

ROOT = Path(__file__).resolve().parents[1]


def load_original(name):
    spec = importlib.util.spec_from_file_location("original_" + name, ROOT / "tests" / "upstream" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def error_metrics(a, b):
    delta = (a.detach().float() - b.detach().float()).cpu()
    norm = float(b.detach().float().cpu().norm())
    return {"max_abs_error": float(delta.abs().max()), "relative_l2_error": float(delta.norm()) / max(norm, 1e-12)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path)
    p.add_argument("--sft")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--toy", action="store_true")
    p.add_argument("--atol", type=float, default=1e-6)
    p.add_argument("--rtol", type=float, default=1e-5)
    args = p.parse_args()
    runtime = None
    if args.toy:
        from flowvla_rl.toy_backend import TinyPolicy, make_chunks
        torch.manual_seed(3)
        policy = TinyPolicy()
        ref = make_frozen_reference(policy.model)
        chunks = make_chunks(policy)
        initial = copy.deepcopy(policy.state_dict())
        optimizer = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=1e-6, weight_decay=1e-10)
        optimizer_state = copy.deepcopy(optimizer.state_dict())
    else:
        if not args.cache or not args.sft or not torch.cuda.is_available():
            p.error("real gate requires --cache, --sft and CUDA")
        from flowvla_rl.libero_rollout import make_runtime
        runtime = make_runtime(args.sft, "libero_object", task_id=0)
        policy = runtime.policy
        ref = make_frozen_reference(policy.model)
        bundle = torch.load(args.cache, map_location="cpu", weights_only=False)
        if state_hash(ref) != bundle["metadata"]["reference_hash"]:
            raise ValueError("reference hash mismatch")
        initial, optimizer_state, chunks = bundle["current"], bundle["optimizer"], bundle["chunks"]
        policy.load_state_dict(initial)
        optimizer = torch.optim.AdamW([x for x in policy.parameters() if x.requires_grad], lr=1e-6, weight_decay=1e-10)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
    device = next(policy.parameters()).device
    rows = []
    report = {"backend": "toy_cpu" if args.toy else "smolvla_cuda", "status": "failed",
              "source_hash": source_hash(ROOT), "atol": args.atol, "rtol": args.rtol,
              "reference_hash": state_hash(ref), "checks": rows}
    try:
        original_score = load_original("smolvla_sde").recompute_log_probs
        for i, stored in enumerate(chunks):
            chunk = stored.to(device)
            with torch.no_grad(), rollout_autocast(device):
                masks, cache = prepare_policy_prefix(policy, chunk.conditioning_batch)
                full = recompute_log_probs(policy.model, masks, cache, chunk.trajectory,
                                           valid_positions=chunk.valid_positions)
                old = chunk.trajectory.masked_log_probs(chunk.valid_positions)
                if float((full-old).abs().max()) > 0.01 or not torch.isfinite(full).all():
                    raise AssertionError("GPU old/current delta exceeds uniform 0.01 gate")
                original = original_score(policy.model, masks, cache, chunk.trajectory,
                                            valid_positions=chunk.valid_positions)
                torch.testing.assert_close(full, original, atol=args.atol, rtol=args.rtol)
                rows.append({"check": "full_vs_upstream", "chunk": i, **error_metrics(full, original)})
                rows.append({"check": "old_vs_current", "chunk": i, **error_metrics(full, old)})
                for method in ("late5", "uniform5"):
                    steps = select_steps(method, root_seed=selection_seed(17), attempted_group_id=0,
                                         episode_id=chunk.episode_id, chunk_id=chunk.chunk_id)
                    subset = recompute_log_probs(policy.model, masks, cache, chunk.trajectory,
                                                  valid_positions=chunk.valid_positions, step_indices=steps)
                    torch.testing.assert_close(subset, full[:, steps], atol=args.atol, rtol=args.rtol)
                    rows.append({"check": method, "chunk": i, **error_metrics(subset, full[:, steps])})
        # Parameter-update regression without keeping two real models resident.
        original_learner = load_original("training").optimize_grpo
        original_learner(policy, ref, chunks, optimizer, clip_epsilon=0.2, kl_beta=0.01,
                         epochs=1, max_grad_norm=1, seed=17)
        expected = {n: p.detach().cpu().clone() for n, p in policy.named_parameters() if p.requires_grad}
        policy.load_state_dict(initial)
        optimizer.load_state_dict(copy.deepcopy(optimizer_state))
        metrics = optimize_grpo(policy, ref, chunks, optimizer, clip_epsilon=0.2, kl_beta=0.01,
                                epochs=1, max_grad_norm=1, seed=17)
        for name, param in policy.named_parameters():
            if name in expected:
                torch.testing.assert_close(param.detach().cpu(), expected[name], atol=args.atol, rtol=args.rtol)
                rows.append({"check": "full_update", "parameter": name, **error_metrics(param.cpu(), expected[name])})
        declared = {n for n, param in policy.named_parameters() if param.requires_grad}
        if declared != set(metrics["actual_gradient_names"]):
            raise AssertionError("declared trainable set differs from actual gradient set")
        report["status"] = "passed"
        report["actual_gradient_names"] = metrics["actual_gradient_names"]
        print(f"{report['backend']} fixed-cache regression gate passed")
    except BaseException as exc:
        report["error"] = repr(exc)
        raise
    finally:
        atomic_json(args.output, report)
        if runtime:
            runtime.close()


if __name__ == "__main__":
    main()
