#!/usr/bin/env python
"""FlowSift v3 online training. Requires the pinned Linux/CUDA/LIBERO runtime."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from flowvla_rl.checkpointing import (RunState, atomic_json, file_hash, read_checkpoint,
                                     restore_checkpoint, save_checkpoint, state_hash)
from flowvla_rl.diagnostics import score_drift
from flowvla_rl.libero_rollout import collect_group, make_runtime
from flowvla_rl.protocol import canonical_hash, load_config, validate_config, validate_lock
from flowvla_rl.provenance import environment_report, model_files, source_hash
from flowvla_rl.reference import make_frozen_reference
from flowvla_rl.runner import run_budget, restore_elapsed_from_log
from flowvla_rl.step_selection import RNG_VERSION, selection_seed, select_steps
from flowvla_rl.training import optimize_grpo, tree_to

ROOT = Path(__file__).resolve().parents[1]


def append_json(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def reconcile_log(run_dir, boundary):
    """Keep interrupted/replayed work in a separate resource ledger, not in main rows."""
    path = run_dir / "groups.jsonl"
    if not path.exists():
        return []
    kept, orphaned = [], []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            orphaned.append({"truncated_log_line": line})
            continue
        (kept if row.get("attempted_groups", boundary + 1) <= boundary else orphaned).append(row)
    if orphaned:
        archive = run_dir / f"replayed-work-{uuid.uuid4().hex}.jsonl"
        archive.write_text("".join(json.dumps(row) + "\n" for row in orphaned))
        temporary = path.with_suffix(".tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in kept))
        os.replace(temporary, path)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft", help="local pinned SFT directory produced by prepare_model.py")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", type=Path, help="checkpoint directory or latest.json")
    parser.add_argument("--protocol-lock", type=Path)
    parser.add_argument("--diagnostic-cache", type=Path, help="trusted torch file containing eight fixed StoredChunks")
    parser.add_argument("--correctness-report", type=Path, help="passed smolvla_cuda report from verify_cache.py")
    parser.add_argument("--export-profile-groups", type=Path, help="export warmed snapshots for three distinct tasks")
    parser.add_argument("--stage", choices=["smoke", "pilot", "formal"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    for arg, key in ((args.stage, "stage"), (args.seed, "seed"), (args.budget, "attempted_budget")):
        if arg is not None:
            config["run"][key] = arg
    validate_config(config)
    lock = None
    if config["run"]["stage"] == "formal":
        if not args.protocol_lock:
            parser.error("formal runs require --protocol-lock")
        lock = json.loads(args.protocol_lock.read_text())
        validate_lock(lock, config)
    if args.validate_only:
        print(json.dumps(config, indent=2))
        return
    if not args.sft or args.run_dir is None:
        parser.error("training requires --sft and --run-dir")
    if not torch.cuda.is_available():
        raise RuntimeError("real LIBERO training requires CUDA; run CPU correctness tests on this host")
    if config["run"]["stage"] != "smoke" and args.diagnostic_cache is None:
        parser.error("pilot/formal require the same fixed --diagnostic-cache (8 chunks)")
    if config["run"]["stage"] != "smoke" and args.correctness_report is None:
        parser.error("pilot/formal require --correctness-report from the server GPU gate")
    sft = Path(args.sft).resolve()
    source = json.loads((sft / "source_manifest.json").read_text())
    for key in ("repo", "revision", "subdirectory"):
        if source[key] != config["model"][key]:
            raise ValueError(f"SFT source {key} differs from locked protocol")
    hashes = model_files(sft)
    if hashes != source["files"]:
        raise ValueError("SFT files changed after download/hash registration")
    seed, method = config["run"]["seed"], config["selection"]["method"]
    if args.export_profile_groups and (method != "full10" or config["run"]["stage"] != "smoke"):
        raise ValueError("profile caches must come from a separate Full-10 Stage-A smoke run")
    root_seed = selection_seed(seed)
    saved = read_checkpoint(args.resume) if args.resume else None
    if args.run_dir.exists() and not saved:
        raise FileExistsError("run-dir already exists; use --resume or a fresh output directory")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if not saved:
        atomic_json(args.run_dir / "run_manifest.json", {"config": config, "method": method,
                    "train_seed": seed, "budget": config["run"]["attempted_budget"],
                    "run_id": args.run_dir.name, "status": "initializing"})
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ.setdefault("MUJOCO_GL", "egl")
    runtime = None
    optimizer = None
    session_start = time.perf_counter()
    state = RunState()
    active_chunks, active_id = [], 0
    exported_tasks = set()
    if args.export_profile_groups:
        args.export_profile_groups.mkdir(parents=True, exist_ok=True)
        for path in args.export_profile_groups.glob("cache_*.json"):
            exported_tasks.add(json.loads(path.read_text())["task_id"])
    try:
        # Construct reference from immutable initial SFT BEFORE loading resumed current.
        runtime = make_runtime(str(sft), "libero_object", task_ids=list(range(10)))
        for key, expected in (("num_steps", 10), ("chunk_size", 50), ("n_action_steps", 50), ("max_action_dim", 32)):
            if getattr(runtime.policy.config, key) != expected:
                raise ValueError(f"SFT config {key} must be {expected}")
        reference = make_frozen_reference(runtime.policy.model)
        reference_hash = state_hash(reference)
        if args.correctness_report:
            gate = json.loads(args.correctness_report.read_text())
            if (gate["status"] != "passed" or gate["backend"] != "smolvla_cuda"
                    or gate["source_hash"] != source_hash(ROOT) or gate["reference_hash"] != reference_hash):
                raise ValueError("GPU correctness report failed or belongs to different source/reference")
        names = [n for n, p in runtime.policy.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([p for p in runtime.policy.parameters() if p.requires_grad],
                                     lr=1e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-10)
        metadata = {"protocol_hash": canonical_hash(config), "source_hash": source_hash(ROOT),
                    "reference_hash": reference_hash, "model_files": hashes,
                    "selection_seed": root_seed, "selection_rng_version": RNG_VERSION,
                    "stable_ids": "attempted_group_id:0-based/episode_id:global/chunk_id:episode-local",
                    "trainable_parameter_names": names,
                    "diagnostic_cache_hash": file_hash(args.diagnostic_cache) if args.diagnostic_cache else None,
                    "protocol_lock_hash": canonical_hash(lock) if lock else None}
        metadata["correctness_report_hash"] = file_hash(args.correctness_report) if args.correctness_report else None
        if lock:
            for key in ("source_hash", "reference_hash", "diagnostic_cache_hash", "correctness_report_hash"):
                if lock[key] != metadata[key]:
                    raise ValueError(f"formal {key} differs from locked Full pilot provenance")
        environment = environment_report(ROOT)
        if config["run"]["stage"] == "formal" and not environment["code_commit"]:
            raise ValueError("formal runs require a recorded FlowSift git commit")
        manifest = {**metadata, **environment, "config": config, "method": method, "train_seed": seed,
                    "budget": config["run"]["attempted_budget"], "model_revision": source["revision"],
                    "sft_path": str(sft), "run_id": args.run_dir.name}
        if saved:
            existing = json.loads((args.run_dir / "run_manifest.json").read_text())
            if any(existing.get(k) != v for k, v in metadata.items()):
                raise ValueError("resume run manifest differs from this run")
            state = restore_checkpoint(saved, runtime.policy, optimizer, metadata)
            # Shared tensors are frozen; verify current loading did not alter reference.
            if state_hash(reference) != reference_hash:
                raise ValueError("resume changed immutable reference tensors")
            kept = reconcile_log(args.run_dir, state.attempted_groups)
            restore_elapsed_from_log(state, kept)
        else:
            atomic_json(args.run_dir / "run_manifest.json", manifest)
            atomic_json(args.run_dir / "resolved_config.json", config)
        fixed = torch.load(args.diagnostic_cache, map_location="cpu", weights_only=False) if args.diagnostic_cache else []
        if fixed and len(fixed) != 8:
            raise ValueError("diagnostic cache must contain exactly eight chunks")

        def export(directory):
            runtime.policy.save_pretrained(directory)
            runtime.preprocessor.save_pretrained(directory)
            runtime.postprocessor.save_pretrained(directory)

        def save(current_state, final):
            return save_checkpoint(args.run_dir / "checkpoints", runtime.policy, optimizer,
                                   current_state, metadata, final=final, export=export)

        def collect(attempt, episode_start):
            nonlocal active_chunks, active_id
            active_id, active_chunks = attempt, []
            task_id, init_id = attempt % 10, (attempt // 10) % 50
            runtime.select_task(task_id)
            reset_seed, policy_seed = seed + attempt, seed + 100_000 + attempt * 4
            runtime.collection_progress = {"completed_episodes": 0, "env_steps": 0}
            episodes, chunks = collect_group(runtime, group_size=4, eta=0.4, init_state_id=init_id,
                                             reset_seed=reset_seed, policy_seed=policy_seed,
                                             episode_id_start=episode_start, chunk_discount=1.0)
            active_chunks = chunks
            return episodes, chunks, {"task_id": task_id, "init_state_id": init_id,
                    "reset_seed": reset_seed, "policy_seeds": list(range(policy_seed, policy_seed + 4)),
                    "initial_observation_digests": [e.initial_observation_digest for e in episodes]}

        def learn(chunks, attempt):
            task_id = attempt % 10
            if args.export_profile_groups and state.completed_updates > 0 and task_id not in exported_tasks and len(exported_tasks) < 3:
                path = args.export_profile_groups / f"cache_task{task_id}_attempt{attempt:06d}.pt"
                bundle = {"chunks": chunks, "current": tree_to(runtime.policy.state_dict(), "cpu"),
                          "optimizer": tree_to(optimizer.state_dict(), "cpu"), "metadata": metadata,
                          "policy_hash": state_hash(runtime.policy), "attempted_group_id": attempt,
                          "task_id": task_id, "train_seed": seed, "sft_path": str(sft)}
                torch.save(bundle, path)
                atomic_json(path.with_suffix(".json"), {"task_id": task_id, "policy_hash": bundle["policy_hash"]})
                exported_tasks.add(task_id)
            return optimize_grpo(runtime.policy, reference, chunks, optimizer,
                                  clip_epsilon=0.2, kl_beta=0.01, epochs=1, max_grad_norm=1,
                                  seed=seed + attempt, method=method, root_seed=root_seed,
                                  attempted_group_id=attempt)

        def diagnose(attempt):
            rows = score_drift(runtime.policy, reference, fixed, attempt)
            path = args.run_dir / "step_diagnostics.csv"
            if rows:
                exists = path.exists()
                with path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    if not exists:
                        writer.writeheader()
                    writer.writerows(rows)

        if not saved:
            save(state, False)  # Recover even if the first group fails.
        summary = run_budget(budget=config["run"]["attempted_budget"], state=state,
                             collect=collect, learn=learn, save=save,
                             log=lambda row: append_json(args.run_dir / "groups.jsonl", row),
                             diagnose=diagnose if fixed else None)
        summary["final_policy_hash"] = state_hash(runtime.policy)
        atomic_json(args.run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    except BaseException as exc:
        failure = {"status": "failed", "error": repr(exc), "attempted_id": active_id,
                   "session_seconds": time.perf_counter() - session_start,
                   "partial_collection": getattr(runtime, "collection_progress", {}),
                   "selection": [{"episode_id": c.episode_id, "chunk_id": c.chunk_id,
                       "steps": select_steps(method, root_seed=root_seed, attempted_group_id=active_id,
                                              episode_id=c.episode_id, chunk_id=c.chunk_id)} for c in active_chunks]}
        if active_chunks:
            replay_path = args.run_dir / f"failure-replay-{uuid.uuid4().hex}.pt"
            # Keep the exact failed learner inputs and current weights. The last
            # periodic checkpoint may precede this cache by several updates.
            torch.save({"chunks": active_chunks,
                        "current": tree_to(runtime.policy.state_dict(), "cpu"),
                        "optimizer": tree_to(optimizer.state_dict(), "cpu") if optimizer else None,
                        "attempted_group_id": active_id, "train_seed": seed,
                        "sft_path": str(sft), "method": method,
                        "purpose": "failure replay; not a complete-group resume boundary"}, replay_path)
            failure["replay_artifact"] = str(replay_path)
        append_json(args.run_dir / "failures.jsonl", failure)
        atomic_json(args.run_dir / "summary.json", failure)
        raise
    finally:
        if runtime is not None:
            runtime.close()
        append_json(args.run_dir / "sessions.jsonl", {"session_seconds": time.perf_counter() - session_start,
                    "resumed": bool(saved), "attempted_groups_at_exit": state.attempted_groups})


if __name__ == "__main__":
    main()
