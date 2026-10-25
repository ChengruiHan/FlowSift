"""Conservative summaries: preserve failed runs and every seed; no equivalence claims."""
from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def profile_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["backend"], row["method"], row["cache_id"])].append(float(row["learner_ms"]))
    medians = {key: statistics.median(values) for key, values in grouped.items()}
    result = []
    for (backend, method, cache), median in sorted(medians.items()):
        baseline = medians.get((backend, "full10", cache))
        measurements = [r for r in rows if (r["backend"], r["method"], r["cache_id"]) == (backend, method, cache)]
        def peak(key):
            values = [int(r[key]) for r in measurements if r.get(key) not in (None, "")]
            return max(values) if values else None
        result.append({"backend": backend, "method": method, "cache_id": cache,
                       "median_learner_ms": median,
                       "time_reduction": 1 - median / baseline if baseline else None,
                       "speedup": baseline / median if baseline else None,
                       "repeats": len(grouped[(backend, method, cache)]),
                       "current_calls_per_chunk": (int(measurements[0]["current_calls"]) / int(measurements[0]["chunk_count"])) if measurements[0].get("chunk_count") else None,
                       "reference_calls_per_chunk": (int(measurements[0]["reference_calls"]) / int(measurements[0]["chunk_count"])) if measurements[0].get("chunk_count") else None,
                       "allocated_peak": peak("allocated_peak"), "reserved_peak": peak("reserved_peak"),
                       "initialization_allocated_peak": peak("initialization_allocated_peak")})
    return result


def task_success(rows, expected_episodes):
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        key = (int(row["task_id"]), int(row["episode_index"]))
        if not 0 <= key[1] < expected_episodes:
            raise ValueError("evaluation episode_index outside the expected range")
        if key in seen:
            raise ValueError("duplicate evaluation episode")
        seen.add(key)
        success = int(row["success"])
        if success not in (0, 1):
            raise ValueError("success must be binary")
        grouped[key[0]].append(success)
    if set(grouped) != set(range(10)) or any(len(v) != expected_episodes for v in grouped.values()):
        raise ValueError("evaluation must cover all ten tasks with the expected episode count")
    return {task: sum(values) / len(values) for task, values in grouped.items()}


def validate_evaluation(rows, manifest, status):
    """Reject stale or mixed CSV/manifest/summary artifacts before aggregation."""
    count = 20 if manifest["split"] == "test" else 10
    seed = 1000 if manifest["split"] == "test" else 2000
    if (manifest.get("episodes_per_task") != count or manifest.get("seed") != seed
            or manifest.get("sampler") != "ode" or not manifest.get("checkpoint_hash")):
        raise ValueError("evaluation manifest differs from the reporting protocol")
    for row in rows:
        if any(row.get(k) != manifest[k] for k in ("run_id", "split", "checkpoint_hash")):
            raise ValueError("evaluation row provenance differs from manifest")
        if int(row["evaluation_seed"]) != seed + int(row["episode_index"]):
            raise ValueError("evaluation seed differs from protocol")
    tasks = task_success(rows, count)
    sr = sum(tasks.values()) / 10
    if (status.get("run_id") != manifest["run_id"] or status.get("episodes") != len(rows)
            or abs(float(status["success_rate"]) - sr) > 1e-12):
        raise ValueError("evaluation summary differs from episode rows")
    return tasks


def summarize_runs(runs_root, evaluations_root=None, split="test"):
    if split not in ("dev", "test"):
        raise ValueError("report split must be dev or test")
    evals, task_rows, costs, online = {}, [], [], []
    evaluation_hashes = {}
    if evaluations_root is not None:
        for path in Path(evaluations_root).glob("*/eval_episodes.csv"):
            status = json.loads((path.parent / "summary.json").read_text())
            manifest = json.loads((path.parent / "eval_manifest.json").read_text())
            if manifest["split"] != split or status["status"] != "complete":
                continue
            run_id = manifest["run_id"]
            if run_id in evals:
                raise ValueError(f"multiple completed {split} evaluations for {run_id}; choose one explicitly")
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            tasks = validate_evaluation(rows, manifest, status)
            evaluation_hashes[run_id] = manifest["checkpoint_hash"]
            evals[run_id] = (sum(tasks.values()) / 10, status["evaluation_seconds"])
            task_rows.extend({"run_id": run_id, "task_id": t, "success_rate": sr} for t, sr in tasks.items())
    if "sft" in evals:
        online.append({"run_id": "sft", "method": "sft", "seed": None, "attempted": 0,
                       "episodes": 0, "updates": 0, "status": "complete", "ode_success_rate": evals["sft"][0],
                       "delta_sft_pp": 0})
    for path in sorted(Path(runs_root).glob("*/run_manifest.json")):
        manifest = json.loads(path.read_text())
        summary_path = path.parent / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {"status": "incomplete"}
        run_id = manifest.get("run_id", path.parent.name)
        if run_id in evals and summary.get("final_policy_hash") != evaluation_hashes[run_id]:
            raise ValueError(f"evaluation checkpoint differs from final policy for {run_id}")
        all_logs = read_jsonl(path.parent / "groups.jsonl")
        groups = [r for r in all_logs if "attempted_id" in r]
        sr = evals.get(run_id, (None, None))[0] if summary["status"] == "complete" else None
        sft = evals.get("sft", (None, None))[0]
        online.append({"run_id": run_id, "method": manifest["method"], "seed": manifest["train_seed"],
                       "stage": manifest["config"]["run"]["stage"], "budget": manifest["budget"],
                       "attempted": summary.get("attempted_groups", len(groups)),
                       "episodes": summary.get("episodes", sum(g["episode_count"] for g in groups)),
                       "env_steps": summary.get("env_steps", sum(g["env_steps"] for g in groups)),
                       "updates": summary.get("completed_updates", sum(g["mixed"] for g in groups)),
                       "status": summary["status"], "ode_success_rate": sr,
                       "delta_sft_pp": (sr-sft)*100 if sr is not None and sft is not None else None})
        cost = {"run_id": run_id, "method": manifest["method"], "seed": manifest["train_seed"],
                "status": summary["status"],
                **{key: sum(float(g.get(key, 0)) for g in all_logs)
                   for key in ("rollout_seconds", "learner_seconds", "diagnostic_seconds", "checkpoint_seconds")},
                "train_elapsed_seconds": summary.get("elapsed_seconds"),
                "allocated_process_seconds": sum(r["session_seconds"] for r in read_jsonl(path.parent / "sessions.jsonl")),
                "evaluation_seconds": evals.get(run_id, (None, None))[1],
                "failed_sessions_seconds": sum(r.get("session_seconds", 0) for r in read_jsonl(path.parent / "failures.jsonl")),
                "replayed_group_count": sum(len([r for r in read_jsonl(p) if "attempted_id" in r])
                                            for p in path.parent.glob("replayed-work-*.jsonl"))}
        measured = sum(cost[k] for k in ("rollout_seconds", "learner_seconds", "diagnostic_seconds", "checkpoint_seconds"))
        cost["other_seconds"] = cost["train_elapsed_seconds"] - measured if cost["train_elapsed_seconds"] is not None else None
        costs.append(cost)
    unknown = set(evals) - {row["run_id"] for row in online}
    if unknown:
        raise ValueError(f"evaluations have no corresponding training run: {sorted(unknown)}")
    # Only compare complete, same-seed, same-stage, same-budget runs.
    baselines = {(r["seed"], r.get("stage"), r.get("budget")): r for r in online if r["method"] == "full10" and r["status"] == "complete"}
    for row in online:
        baseline = baselines.get((row["seed"], row.get("stage"), row.get("budget")))
        a, b = row.get("ode_success_rate"), baseline.get("ode_success_rate") if baseline else None
        row["delta_full_pp"] = (a-b)*100 if a is not None and b is not None else None
    return online, costs, task_rows
