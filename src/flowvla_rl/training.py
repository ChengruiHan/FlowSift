"""FlowSift episode-balanced, single-epoch selective GRPO learner."""
from __future__ import annotations

import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from flowvla_rl.grpo import grpo_loss
from flowvla_rl.step_selection import select_steps, selection_seed
from flowvla_rl.smolvla_sde import (
    SmolVLATrajectory, prepare_policy_prefix, recompute_log_probs, rollout_autocast,
)


def tree_to(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, Tensor):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to(item, device) for item in value)
    return value


@dataclass(frozen=True)
class StoredChunk:
    episode_id: int
    conditioning_batch: dict[str, Any]
    trajectory: SmolVLATrajectory
    valid_positions: Tensor
    advantage: float
    chunk_id: int = 0

    def to(self, device: torch.device | str) -> "StoredChunk":
        return StoredChunk(self.episode_id, tree_to(self.conditioning_batch, device),
                           self.trajectory.to(device), self.valid_positions.to(device),
                           self.advantage, self.chunk_id)


def episode_balanced_chunk_weights(chunks: list[StoredChunk]) -> list[float]:
    if not chunks:
        raise ValueError("at least one rollout chunk is required")
    counts = Counter(chunk.episode_id for chunk in chunks)
    return [1.0 / (len(counts) * counts[chunk.episode_id]) for chunk in chunks]


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class PhaseTimer:
    """Optional separate profiling pass: CUDA events, no per-phase synchronization."""
    def __init__(self, device, enabled=False):
        self.device, self.enabled, self.records = device, enabled, []

    def start(self, phase):
        if not self.enabled:
            return None
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return phase, event
        return phase, time.perf_counter()

    def stop(self, token):
        if token is None:
            return
        if self.device.type == "cuda":
            end = torch.cuda.Event(enable_timing=True)
            end.record()
        else:
            end = time.perf_counter()
        self.records.append((*token, end))

    def results(self):
        result = {}
        for phase, start, end in self.records:
            seconds = start.elapsed_time(end) / 1000 if self.device.type == "cuda" else end - start
            result[phase + "_seconds"] = result.get(phase + "_seconds", 0) + seconds
        return result


def optimize_grpo(policy: nn.Module, reference_model: nn.Module,
                  chunks: list[StoredChunk], optimizer: torch.optim.Optimizer, *,
                  clip_epsilon: float, kl_beta: float, epochs: int,
                  max_grad_norm: float, seed: int, method: str = "full10",
                  root_seed: int | None = None, attempted_group_id: int = 0,
                  initial_logp_tolerance: float = 0.01,
                  detailed_timing: bool = False) -> dict:
    if not chunks or epochs != 1 or max_grad_norm <= 0:
        raise ValueError("v3 requires nonempty chunks, epochs=1 and positive grad norm")
    if len({(c.episode_id, c.chunk_id) for c in chunks}) != len(chunks):
        raise ValueError("episode/chunk IDs must be unique within an update")
    device = next(policy.parameters()).device
    # eval disables dropout, but does not disable autograd. Rollout uses this mode too.
    policy.eval()
    reference_model.eval()
    optimizer.zero_grad(set_to_none=True)
    root_seed = selection_seed(seed) if root_seed is None else root_seed
    order = list(range(len(chunks)))
    random.Random(seed).shuffle(order)
    weights = episode_balanced_chunk_weights(chunks)
    totals = dict.fromkeys(("loss", "pg_loss", "kl", "ratio_mean", "clip_fraction"), 0.0)
    max_delta, max_d, clamp_hits, elements = 0.0, float("-inf"), 0, 0
    calls = {"current_calls": 0, "reference_calls": 0}
    timer = PhaseTimer(device, detailed_timing)
    synchronize(device)
    started = time.perf_counter()
    try:
        for index in order:
            token = timer.start("transfer")
            chunk = chunks[index].to(device)
            timer.stop(token)
            if chunk.trajectory.states.shape[0] != 1:
                raise ValueError("v3 learner_chunk_batch_size must be 1")
            steps = select_steps(method, root_seed=root_seed,
                                 attempted_group_id=attempted_group_id,
                                 episode_id=chunk.episode_id, chunk_id=chunk.chunk_id,
                                 steps=policy.model.config.num_steps)
            with rollout_autocast(device):
                with torch.no_grad():
                    token = timer.start("prefix")
                    masks, cache = prepare_policy_prefix(policy, chunk.conditioning_batch)
                    timer.stop(token)
                    token = timer.start("reference")
                    reference_logp = recompute_log_probs(
                        reference_model, masks, cache, chunk.trajectory,
                        valid_positions=chunk.valid_positions, step_indices=steps)
                    calls["reference_calls"] += len(steps)
                    timer.stop(token)
                token = timer.start("current")
                current_logp = recompute_log_probs(
                    policy.model, masks, cache, chunk.trajectory,
                    valid_positions=chunk.valid_positions, step_indices=steps)
                calls["current_calls"] += len(steps)
                timer.stop(token)
            old_logp = chunk.trajectory.masked_log_probs(chunk.valid_positions).detach()[:, steps]
            delta = current_logp.detach().float() - old_logp.float()
            d = reference_logp.detach().float() - current_logp.detach().float()
            if not all(torch.isfinite(x).all() for x in (current_logp, reference_logp, old_logp)):
                raise FloatingPointError(f"nonfinite logp at chunk {(chunk.episode_id, chunk.chunk_id)}, steps={steps}")
            chunk_delta = float(delta.abs().max())
            max_delta = max(max_delta, chunk_delta)
            if chunk_delta > initial_logp_tolerance:
                raise RuntimeError(f"old/current mismatch: abs(delta_logp)={chunk_delta:.6g} "
                                   f"> {initial_logp_tolerance}; chunk={(chunk.episode_id, chunk.chunk_id)}, steps={steps}")
            max_d = max(max_d, float(d.max()))
            clamp_hits += int((delta.abs() > 20).sum())
            elements += delta.numel()
            token = timer.start("loss_backward")
            loss, metrics = grpo_loss(
                current_logp[:, None, :], old_logp[:, None, :], reference_logp[:, None, :],
                torch.tensor([chunk.advantage], device=device),
                clip_epsilon=clip_epsilon, kl_beta=kl_beta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite k3/loss: max_d={max_d}, steps={steps}")
            (loss * weights[index]).backward()
            timer.stop(token)
            for key in totals:
                totals[key] += float(metrics[key]) * weights[index]
        token = timer.start("optimizer")
        actual_grad_names = [name for name, p in policy.named_parameters() if p.grad is not None]
        trainable = [p for p in policy.parameters() if p.requires_grad]
        if not actual_grad_names:
            raise RuntimeError("learner produced no parameter gradients")
        norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        timer.stop(token)
        synchronize(device)
        elapsed = time.perf_counter() - started
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {**totals, **calls, **timer.results(), "grad_norm": float(norm),
            "grad_norm_pre_clip": float(norm), "gradient_clipped": bool(norm > max_grad_norm),
            "actual_gradient_names": actual_grad_names, "optimizer_steps": 1,
            "chunks": len(chunks), "learner_seconds": elapsed,
            "max_abs_initial_delta_logp": max_delta, "raw_ratio_clamp_fraction": clamp_hits / elements,
            "max_k3_d": max_d, "nonfinite_count": 0, "k3_loss": totals["kl"]}
