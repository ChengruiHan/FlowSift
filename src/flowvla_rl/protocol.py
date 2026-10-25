"""Locked protocol validation and immutable experiment planning."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml

from flowvla_rl.step_selection import METHODS

UPSTREAM_COMMIT = "50e73afc07212f48bed370ea4c88a3131d6f914a"
MODEL_REPO = "MorpheusTzz/smolvla-grpo-libero-object"
MODEL_REVISION = "a120aa9b51f8e55db5ad8284e051963c61a2f5ed"
MODEL_SUBDIRECTORY = "sft-100pct-baseline"


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def load_config(path) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    validate_config(config)
    return config


def validate_config(c):
    required = {
        "protocol_version": "v3.0", "upstream.commit": UPSTREAM_COMMIT,
        "model.repo": MODEL_REPO, "model.revision": MODEL_REVISION,
        "model.subdirectory": MODEL_SUBDIRECTORY, "model.denoising_steps": 10,
        "model.chunk_size": 50, "model.n_action_steps": 50,
        "model.freeze_state_projection": True,
        "model.reference_source": "immutable_initial_sft",
        "train.suite": "libero_object", "train.task_ids": list(range(10)),
        "train.group_size": 4, "train.groups_per_update": 1, "train.eta": 0.4,
        "train.learning_rate": 1e-6, "train.optimizer": "AdamW",
        "train.betas": [0.9, 0.999], "train.eps": 1e-8, "train.weight_decay": 1e-10,
        "train.clip_epsilon": 0.2, "train.kl_beta": 0.01, "train.epochs": 1,
        "train.max_grad_norm": 1.0, "train.chunk_discount": 1.0,
        "train.advantage_std_normalization": False, "train.learner_chunk_batch_size": 1,
        "train.budget_unit": "attempted_groups", "train.collapsed_group_early_stop": False,
        "selection.uniform_count": 5, "selection.sample_per": "chunk",
        "selection.without_replacement": True, "selection.independent_keyed_rng": True,
        "selection.retain_original_time_grid": True, "selection.normalize_by_selected_count": True,
        "evaluation.sampler": "ode", "evaluation.denoising_steps": 10,
        "evaluation.max_env_steps": 280, "evaluation.batch_size": 1,
        "evaluation.dev_seed": 2000, "evaluation.dev_episodes_per_task": 10,
        "evaluation.test_seed": 1000, "evaluation.test_episodes_per_task": 20,
        "evaluation.checkpoint": "final", "evaluation.strict_held_out_claim": False,
        "diagnostics.every_attempted_groups": 50, "diagnostics.fixed_chunks": 8,
        "diagnostics.score_all_10_steps": True,
        "checkpoint.every_attempted_groups": 50, "checkpoint.always_save_final": True,
    }
    for dotted, expected in required.items():
        value = c
        try:
            for key in dotted.split("."):
                value = value[key]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing protocol field {dotted}") from exc
        if value != expected:
            raise ValueError(f"{dotted} must be {expected!r}; got {value!r}")
    if c["selection"]["method"] not in METHODS:
        raise ValueError("unsupported selection method")
    run = c["run"]
    if run["stage"] not in ("smoke", "pilot", "formal"):
        raise ValueError("stage must be smoke/pilot/formal")
    if type(run["seed"]) is not int or run["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    b = run["attempted_budget"]
    if type(b) is not int or b <= 0:
        raise ValueError("attempted_budget must be a positive integer")
    if run["stage"] == "pilot" and (b != 100 or run["seed"] != 17):
        raise ValueError("pilot requires B=100 and seed=17")
    if run["stage"] == "formal" and (b not in (300, 400, 500, 600) or run["seed"] not in (101, 202, 303)):
        raise ValueError("formal budget/seed outside protocol")


def planned_budget(mixed_groups: int, attempted: int = 100) -> int:
    if attempted != 100 or not 0 <= mixed_groups <= attempted:
        raise ValueError("budget planning requires a complete 100-group Full pilot")
    if mixed_groups == 0:
        raise ValueError("Full pilot has no mixed groups; audit before formal expansion")
    p = mixed_groups / attempted
    return min(600, max(300, 100 * math.ceil(200 / (100 * max(p, 0.1)))))


def validate_lock(lock, config):
    if lock["protocol_version"] != "v3.0" or lock["methods"] != list(METHODS):
        raise ValueError("invalid formal protocol lock")
    if config["run"]["seed"] not in lock["seeds"] or config["run"]["attempted_budget"] != lock["attempted_budget"]:
        raise ValueError("run differs from locked experiment matrix")


def validate_test_matrix(runs_root, lock, run_id):
    """Require both a complete formal matrix and membership of the chosen run."""
    required = {(seed, method) for seed in lock["seeds"] for method in lock["methods"]}
    available, eligible = set(), set()
    for path in Path(runs_root).glob("*/run_manifest.json"):
        manifest = json.loads(path.read_text())
        summary_path = path.parent / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        pair = (manifest["train_seed"], manifest["method"])
        if (summary["status"] == "complete"
                and summary["attempted_groups"] == lock["attempted_budget"]
                and manifest["config"]["run"]["stage"] == "formal"
                and manifest.get("protocol_lock_hash") == canonical_hash(lock)
                and pair in required):
            available.add(pair)
            eligible.add(path.parent.name)
    if not required <= available:
        raise ValueError(f"formal matrix incomplete: {required - available}")
    if run_id != "sft" and run_id not in eligible:
        raise ValueError("test run-id is not a completed member of the locked formal matrix")
