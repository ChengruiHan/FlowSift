#!/usr/bin/env python
"""Logged wrapper around LeRobot's official deterministic LIBERO evaluator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from flowvla_rl.run_logging import RunLedger


CAMERA_MAPPING = '{"agentview_image":"camera1","robot0_eye_in_hand_image":"camera2"}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="lerobot/smolvla_libero")
    parser.add_argument(
        "--suite",
        choices=["libero_object", "libero_spatial", "libero_goal"],
        default="libero_object",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--run-root", default=os.environ.get("FLOWVLA_RUNS", "results/runs")
    )
    args = parser.parse_args()
    if args.episodes <= 0 or not 0 < args.batch_size <= args.episodes:
        raise ValueError("require episodes > 0 and 0 < batch_size <= episodes")

    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_eval",
        f"--policy.path={args.checkpoint}",
        "--env.type=libero",
        f"--env.task={args.suite}",
        f"--eval.batch_size={args.batch_size}",
        f"--eval.n_episodes={args.episodes}",
        f"--eval.use_async_envs={'true' if args.batch_size > 1 else 'false'}",
        "--policy.device=cuda",
        f"--env.camera_name_mapping={CAMERA_MAPPING}",
        "--policy.empty_cameras=1",
        f"--seed={args.seed}",
    ]
    ledger = RunLedger(
        args.run_root,
        f"eval_{args.suite}",
        vars(args),
        project_dir=Path(__file__).parents[1],
        command=" ".join(command),
    )
    output_dir = ledger.path / "eval_artifacts"
    command.append(f"--output_dir={output_dir}")
    (ledger.path / "command.txt").write_text(" ".join(command) + "\n")
    environment = os.environ.copy()
    environment.update(
        {
            "MUJOCO_GL": "egl",
            "HF_HOME": environment.get(
                "HF_HOME", os.path.expanduser("~/.cache/huggingface")
            ),
            "HF_HUB_DOWNLOAD_TIMEOUT": "300",
        }
    )
    try:
        with (ledger.path / "train.log").open("w") as log:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
        eval_info = json.loads((output_dir / "eval_info.json").read_text())
        overall = eval_info["overall"]
        ledger.log_metrics(
            {
                "seed": args.seed,
                "episodes_per_task": args.episodes,
                "avg_sum_reward": float(overall["avg_sum_reward"]),
                "pc_success": float(overall["pc_success"]),
                "eval_s": float(overall["eval_s"]),
            }
        )
        ledger.set_status("complete", eval_info="eval_artifacts/eval_info.json")
        print(f"run_dir={ledger.path}")
    except BaseException as exc:
        ledger.set_status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
