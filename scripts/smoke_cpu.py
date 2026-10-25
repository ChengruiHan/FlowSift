#!/usr/bin/env python
"""Exercise all three learners, collapsed budgets and checkpoint output on CPU."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import torch
from flowvla_rl.checkpointing import RunState, atomic_json, save_checkpoint, state_hash
from flowvla_rl.reference import make_frozen_reference
from flowvla_rl.runner import run_budget
from flowvla_rl.step_selection import METHODS
from flowvla_rl.toy_backend import TinyPolicy, make_chunks
from flowvla_rl.training import optimize_grpo

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/cpu-smoke"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for method in METHODS:
        torch.manual_seed(17)
        policy = TinyPolicy()
        ref = make_frozen_reference(policy.model)
        opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-6)
        logs = []
        def collect(attempt, episode):
            episodes = [SimpleNamespace(success=attempt < 2 and i % 2 == 0, steps=4) for i in range(4)]
            return episodes, make_chunks(policy, 100 + attempt), {}
        summary = run_budget(budget=3, state=RunState(), collect=collect,
            learn=lambda chunks, i: optimize_grpo(policy, ref, chunks, opt, method=method,
                seed=17, attempted_group_id=i, clip_epsilon=0.2, kl_beta=0.01, epochs=1, max_grad_norm=1),
            save=lambda state, final: save_checkpoint(args.output / method, policy, opt, state,
                {"method": method, "reference_hash": state_hash(ref)}, final=final), log=logs.append)
        results.append({"backend": "toy_cpu", "method": method, **summary})
    atomic_json(args.output / "summary.json", results)
    print(f"All three CPU integration runs completed: {args.output}. No LIBERO task results were measured.")
