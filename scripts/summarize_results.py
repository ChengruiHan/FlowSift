#!/usr/bin/env python
"""Generate per-seed tables and plots without mixing CPU demos into GPU evidence."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

from flowvla_rl.reporting import profile_summary, summarize_runs, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--evaluations", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output", type=Path, default=Path("results/report"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    online, costs, tasks = summarize_runs(args.runs, args.evaluations, args.split)
    for name, rows in (("online_per_seed.csv", online), ("system_cost.csv", costs), ("task_success.csv", tasks)):
        write_csv(args.output / name, rows)
    profiles = []
    if args.profile:
        with args.profile.open() as handle:
            profiles = profile_summary(list(csv.DictReader(handle)))
        write_csv(args.output / "profile_paired.csv", profiles)
    if online:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if costs:
            fig, ax = plt.subplots(figsize=(10, 5))
            bottom = [0.0] * len(costs)
            for key in ("rollout_seconds", "learner_seconds", "diagnostic_seconds", "checkpoint_seconds", "other_seconds"):
                values = [r.get(key) or 0 for r in costs]
                ax.bar([r["run_id"] for r in costs], values, bottom=bottom, label=key)
                bottom = [a+b for a,b in zip(bottom, values)]
            ax.set_ylabel("Seconds")
            ax.tick_params(axis="x", rotation=35)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(args.output / "training_time.png", dpi=180)
            plt.close(fig)
        gpu_profiles = [r for r in profiles if r["backend"] == "smolvla_cuda"]
        averages = defaultdict(list)
        for r in gpu_profiles:
            averages[r["method"]].append(r["median_learner_ms"])
        plotted = [r for r in online if r.get("ode_success_rate") is not None and r["method"] in averages]
        if plotted:
            fig, ax = plt.subplots()
            for r in plotted:
                ax.scatter(sum(averages[r["method"]])/len(averages[r["method"]]), r["ode_success_rate"]*100,
                           label=f'{r["method"]} seed {r["seed"]}')
            ax.set_xlabel("Fixed-cache learner ms/update (not inference)")
            ax.set_ylabel("Final ODE success (%)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.output / "success_vs_learner.png", dpi=180)
            plt.close(fig)
        mapping = {r["run_id"]: r for r in online}
        task_map = {(r["run_id"], r["task_id"]): r["success_rate"] for r in tasks}
        deltas = []
        for r in online:
            if r["method"] not in ("late5", "uniform5") or r["status"] != "complete":
                continue
            full = next((b for b in online if b["method"] == "full10" and b["seed"] == r["seed"]
                         and b.get("budget") == r.get("budget") and b.get("stage") == r.get("stage")), None)
            if full and all((x["run_id"], t) in task_map for x in (full, r) for t in range(10)):
                deltas.append((r, [(task_map[r["run_id"], t] - task_map[full["run_id"], t])*100 for t in range(10)]))
        if deltas:
            fig, ax = plt.subplots()
            for r, values in deltas:
                ax.plot(range(10), values, marker="o", label=f'{r["method"]} seed {r["seed"]}')
            ax.axhline(0, color="gray", linewidth=1)
            ax.set_xlabel("LIBERO Object task")
            ax.set_ylabel("Success difference vs same-seed Full (pp)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(args.output / "task_differences.png", dpi=180)
            plt.close(fig)
    print(f"Tables written to {args.output}; missing evaluations remain empty, failed runs remain listed.")


if __name__ == "__main__":
    main()
