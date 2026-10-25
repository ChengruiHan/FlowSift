#!/usr/bin/env python
"""Sequential three-method launch on one GPU, with automatic complete-boundary resume."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from flowvla_rl.step_selection import METHODS

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pilot", "formal"], required=True)
    p.add_argument("--sft", required=True)
    p.add_argument("--diagnostic-cache", required=True)
    p.add_argument("--correctness-report", required=True)
    p.add_argument("--protocol-lock")
    p.add_argument("--run-root", type=Path, default=Path("results/runs"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.stage == "pilot":
        runs, budget = [(17, m) for m in METHODS], 100
    else:
        if not args.protocol_lock:
            p.error("formal requires --protocol-lock")
        lock = json.loads(Path(args.protocol_lock).read_text())
        runs, budget = lock["run_order"], lock["attempted_budget"]
    for seed, method in runs:
        run = args.run_root / f"{args.stage}_{method}_seed{seed}"
        if (run / "summary.json").exists():
            summary = json.loads((run / "summary.json").read_text())
            if summary["status"] == "complete" and summary["attempted_groups"] == budget:
                print(f"Already complete: {run}")
                continue
        command = [sys.executable, str(Path(__file__).with_name("train_grpo_libero.py")),
                   "--config", f"configs/{method}.yaml", "--stage", args.stage,
                   "--seed", str(seed), "--budget", str(budget), "--sft", args.sft,
                   "--run-dir", str(run), "--diagnostic-cache", args.diagnostic_cache,
                   "--correctness-report", args.correctness_report]
        if args.protocol_lock:
            command += ["--protocol-lock", args.protocol_lock]
        latest = run / "checkpoints" / "latest.json"
        if latest.exists():
            command += ["--resume", str(latest)]
        print(__import__("shlex").join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
