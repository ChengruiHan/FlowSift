#!/usr/bin/env python
"""Official LeRobot ODE evaluation with per-episode metadata, same seeds for all runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from flowvla_rl.checkpointing import atomic_json, state_hash
from flowvla_rl.libero_rollout import make_runtime, observation_digest
from flowvla_rl.protocol import validate_test_matrix, MODEL_REPO, MODEL_REVISION, MODEL_SUBDIRECTORY
from flowvla_rl.provenance import model_files


class ObservedEnv:
    """Read-only metadata hooks around official evaluator reset/step calls."""
    def __init__(self, env):
        self.env = env
        self.steps = 0
        self.digest = None
        self.init_state_id = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.steps = 0
        self.digest = observation_digest(observation)
        self.init_state_id = getattr(self.env.envs[0].unwrapped, "init_state_id", None)
        return observation, info

    def step(self, action):
        self.steps += 1
        return self.env.step(action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-id", required=True, help="training run directory name, or sft")
    parser.add_argument("--split", choices=["dev", "test", "replay"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrix-runs", type=Path, help="required for test; contains all completed formal runs")
    parser.add_argument("--protocol-lock", type=Path)
    args = parser.parse_args()
    if args.split == "test":
        if not args.protocol_lock or not args.matrix_runs:
            parser.error("final test requires the protocol lock and completed matrix directory")
        lock = json.loads(args.protocol_lock.read_text())
        validate_test_matrix(args.matrix_runs, lock, args.run_id)
    if not torch.cuda.is_available():
        raise RuntimeError("LIBERO evaluation requires the configured CUDA runtime")
    from lerobot.scripts.lerobot_eval import eval_policy
    os.environ.setdefault("MUJOCO_GL", "egl")
    args.output.mkdir(parents=True, exist_ok=False)
    seed, count = (2000, 10) if args.split == "dev" else (1000, 20 if args.split == "test" else 1)
    runtime = None
    started = time.perf_counter()
    try:
        runtime = make_runtime(args.checkpoint, "libero_object", task_ids=list(range(10)))
        if runtime.policy.config.num_steps != 10:
            raise ValueError("evaluation requires ten-step ODE")
        checkpoint_hash = state_hash(runtime.policy)
        if args.run_id == "sft":
            source = json.loads((Path(args.checkpoint) / "source_manifest.json").read_text())
            if ((source["repo"], source["revision"], source["subdirectory"]) !=
                    (MODEL_REPO, MODEL_REVISION, MODEL_SUBDIRECTORY)
                    or source["files"] != model_files(args.checkpoint)):
                raise ValueError("SFT evaluation source or files differ from protocol")
        elif args.split == "test":
            training_summary = json.loads((args.matrix_runs / args.run_id / "summary.json").read_text())
            if training_summary.get("final_policy_hash") != checkpoint_hash:
                raise ValueError("evaluation checkpoint is not the recorded final policy for this run-id")
        manifest = {"run_id": args.run_id, "checkpoint_hash": checkpoint_hash,
                    "checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
                    "seed": seed, "episodes_per_task": count, "sampler": "ode", "batch_size": 1,
                    "max_env_steps": 280, "strict_held_out_claim": False,
                    "command": __import__("sys").argv, "policy_rng": "seed+episode_index; reset each episode"}
        atomic_json(args.output / "eval_manifest.json", manifest)
        rows = []
        with (args.output / "eval_episodes.csv").open("w", newline="") as handle:
            writer = None
            for task in range(10):
                runtime.select_task(task)
                env = ObservedEnv(runtime.env)
                if runtime.env.call("_max_episode_steps")[0] != 280:
                    raise ValueError("Object episode horizon must be 280")
                for episode in range(count):
                    evaluation_seed = seed + episode
                    random.seed(evaluation_seed)
                    np.random.seed(evaluation_seed)
                    torch.manual_seed(evaluation_seed)
                    result = eval_policy(env=env, policy=runtime.policy,
                        env_preprocessor=runtime.env_preprocessor, env_postprocessor=runtime.env_postprocessor,
                        preprocessor=runtime.preprocessor, postprocessor=runtime.postprocessor,
                        n_episodes=1, start_seed=evaluation_seed, max_episodes_rendered=0)
                    row = {"run_id": args.run_id, "checkpoint_hash": checkpoint_hash, "split": args.split,
                           "task_id": task, "episode_index": episode, "evaluation_seed": evaluation_seed,
                           "init_state_id_if_available": env.init_state_id,
                           "initial_observation_digest_if_available": env.digest,
                           "success": int(result["per_episode"][0]["success"]), "env_steps": env.steps}
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    rows.append(row)
        atomic_json(args.output / "summary.json", {"status": "complete", "run_id": args.run_id,
                    "success_rate": sum(r["success"] for r in rows) / len(rows),
                    "episodes": len(rows), "evaluation_seconds": time.perf_counter() - started})
    except BaseException as exc:
        atomic_json(args.output / "summary.json", {"status": "failed", "error": repr(exc),
                    "evaluation_seconds": time.perf_counter() - started})
        raise
    finally:
        if runtime:
            runtime.close()


if __name__ == "__main__":
    main()
