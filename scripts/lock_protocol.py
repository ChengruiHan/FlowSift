#!/usr/bin/env python
"""Lock the complete three-method matrix before opening final test results."""
import argparse
import json
from pathlib import Path

from flowvla_rl.protocol import planned_budget
from flowvla_rl.step_selection import METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-pilot", type=Path, required=True)
    parser.add_argument("--seeds", choices=["101", "101,202", "101,202,303"], default="101,202")
    parser.add_argument("--resource-budget-300-reason")
    parser.add_argument("--output", type=Path, default=Path("protocol.lock.json"))
    args = parser.parse_args()
    summary = json.loads((args.full_pilot / "summary.json").read_text())
    manifest = json.loads((args.full_pilot / "run_manifest.json").read_text())
    if summary["status"] != "complete" or manifest["method"] != "full10" or manifest["train_seed"] != 17:
        raise ValueError("requires completed Full-10 pilot seed 17")
    suggested = planned_budget(summary["completed_updates"], summary["attempted_groups"])
    seeds = [int(x) for x in args.seeds.split(",")]
    if args.resource_budget_300_reason and seeds != [101]:
        raise ValueError("resource fallback B=300 requires the minimum single-seed matrix")
    lock = {"protocol_version": "v3.0", "methods": list(METHODS), "seeds": seeds,
            **{key: manifest[key] for key in ("source_hash", "reference_hash", "diagnostic_cache_hash", "correctness_report_hash")},
            "attempted_budget": 300 if args.resource_budget_300_reason else suggested,
            "suggested_budget": suggested, "planned_target_updates": 200,
            "resource_reason": args.resource_budget_300_reason,
            "pilot": str(args.full_pilot.resolve()), "test_opened": False,
            "run_order": [[seed, method] for i, seed in enumerate(seeds)
                          for method in (METHODS[i:] + METHODS[:i])]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(lock, handle, indent=2)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
