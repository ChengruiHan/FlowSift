"""Memory-bounded SmolVLA GRPO update over collected action chunks."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from flowvla_rl.grpo import grpo_loss
from flowvla_rl.smolvla_sde import (
    SmolVLATrajectory,
    prepare_policy_prefix,
    recompute_log_probs,
    rollout_autocast,
)


# 首次前向的 ratio 容差。精度对齐后实测 Δlogp 恰好为 0（ratio 恰好 1.0），
# 所以 0.05 已经非常宽松；真正的错配会给出 0.34 这种量级，一眼可辨。
_RATIO_TOLERANCE = 0.05


def tree_to(value: Any, device: torch.device | str) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: tree_to(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_to(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_to(item, device) for item in value)
    return value


@dataclass(frozen=True)
class StoredChunk:
    """A rollout chunk with one episode-level scalar advantage."""

    episode_id: int
    conditioning_batch: dict[str, Any]
    trajectory: SmolVLATrajectory
    valid_positions: Tensor
    advantage: float

    def to(self, device: torch.device | str) -> "StoredChunk":
        return StoredChunk(
            episode_id=self.episode_id,
            conditioning_batch=tree_to(self.conditioning_batch, device),
            trajectory=self.trajectory.to(device),
            valid_positions=self.valid_positions.to(device),
            advantage=self.advantage,
        )


def episode_balanced_chunk_weights(chunks: list[StoredChunk]) -> list[float]:
    """Give every episode equal mass, then divide it across its chunks."""
    if not chunks:
        raise ValueError("at least one rollout chunk is required")
    counts = Counter(chunk.episode_id for chunk in chunks)
    episode_count = len(counts)
    return [1.0 / (episode_count * counts[chunk.episode_id]) for chunk in chunks]


def _optimizer_step(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float,
) -> float:
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm.detach().cpu())


def optimize_grpo(
    policy: nn.Module,
    reference_model: nn.Module,
    chunks: list[StoredChunk],
    optimizer: torch.optim.Optimizer,
    *,
    clip_epsilon: float,
    kl_beta: float,
    epochs: int,
    max_grad_norm: float,
    seed: int,
) -> dict[str, float]:
    """Rescore fixed trajectories and update only the current action expert.

    One chunk is processed at a time to keep the ten denoising graphs bounded in
    memory. The old log-prob is read from rollout storage; no old model copy is
    resident. The reference shares the frozen VLM and is evaluated without grad.
    """
    if not chunks:
        raise ValueError("at least one rollout chunk is required")
    if epochs <= 0 or max_grad_norm <= 0:
        raise ValueError("epochs and max_grad_norm must be positive")
    device = next(policy.parameters()).device
    policy.train()
    reference_model.eval()
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(seed)
    totals = {
        "loss": 0.0,
        "pg_loss": 0.0,
        "kl": 0.0,
        "ratio_mean": 0.0,
        "clip_fraction": 0.0,
    }
    optimizer_steps = 0
    grad_norm_total = 0.0
    checked_ratio = False
    order = list(range(len(chunks)))
    chunk_weights = episode_balanced_chunk_weights(chunks)

    for _epoch in range(epochs):
        rng.shuffle(order)
        # Backward one chunk at a time, but step only after the exact
        # episode-balanced objective has been accumulated. This keeps memory
        # bounded without giving longer failed episodes more weight merely
        # because they contain more control chunks.
        for chunk_index in order:
            chunk = chunks[chunk_index].to(device)
            # 打分必须与**采集**走同一条数值路径。采集在
            # libero_rollout.collect_episode 里是 autocast(bfloat16)；若这里用 fp32
            # 重算，同一条轨迹会算出不同的 log-prob —— 实测 Δlogp = -1.07，
            # 即 ratio = 0.34，完全落在 clip 区间 [1-ε, 1+ε] 之外，于是 PPO 的
            # min() 恒选被 clip 的那一支，梯度退化成噪声，训练看起来"在跑"但学不动
            # （headline run#1 白跑了 75 次更新才发现）。
            # 精度对齐后同一条轨迹逐位复现，Δlogp 恰好为 0、ratio 恰好为 1。
            # prefix 也必须在同一 autocast 下算 —— 它喂给打分，混合精度同样会失配。
            with rollout_autocast(device):
                with torch.no_grad():
                    prefix_masks, prefix_cache = prepare_policy_prefix(
                        policy, chunk.conditioning_batch
                    )
                    reference_logp = recompute_log_probs(
                        reference_model,
                        prefix_masks,
                        prefix_cache,
                        chunk.trajectory,
                        valid_positions=chunk.valid_positions,
                    )
                current_logp = recompute_log_probs(
                    policy.model,
                    prefix_masks,
                    prefix_cache,
                    chunk.trajectory,
                    valid_positions=chunk.valid_positions,
                )
            old_logp = chunk.trajectory.masked_log_probs(chunk.valid_positions)
            advantage = torch.tensor([chunk.advantage], device=device)
            loss, metrics = grpo_loss(
                current_logp[:, None, :],
                old_logp[:, None, :],
                reference_logp[:, None, :],
                advantage,
                clip_epsilon=clip_epsilon,
                kl_beta=kl_beta,
            )
            # 不变量：进入本函数时参数还等于采集时的参数，所以**第一个 chunk 的第一次
            # 前向**必须逐位复现采集时的 log-prob —— ratio 恒为 1。
            # 单元测试抓不到它的违反：测试跑在 CPU 上、autocast 被禁用，而错配只在
            # CUDA 上出现。所以只能在运行时守。
            # 这条守卫的由来：采集用 autocast(bf16)、打分用 fp32，同一条轨迹
            # Δlogp = -1.07 即 ratio = 0.34，完全落在 clip 区间外，PPO 的 min()
            # 恒选被 clip 的那一支，梯度退化成噪声 —— 训练"在跑"、loss 有数、
            # 崩溃率正常，但成功率 75 次更新纹丝不动，白烧 1.4 GPU 小时才被发现。
            if optimizer_steps == 0 and _epoch == 0 and not checked_ratio:
                checked_ratio = True
                drift = abs(float(metrics["ratio_mean"].cpu()) - 1.0)
                if drift > _RATIO_TOLERANCE:
                    raise RuntimeError(
                        f"重打分未能复现采集时的 log-prob：首次前向 ratio_mean 偏离 1 达 {drift:.4f}"
                        f"（容差 {_RATIO_TOLERANCE}）。此时参数尚未更新，ratio 必须为 1。\n"
                        "最可能的原因是采集与打分的数值路径不一致——两边都必须走 "
                        "flowvla_rl.smolvla_sde.rollout_autocast；其次检查 dropout/BN 的 "
                        "train/eval 模式、以及 prefix cache 是否在同一 autocast 下计算。\n"
                        "带着这个偏差继续训练不会报错，但 PPO 目标已失效，只会白烧 GPU。"
                    )
            weight = chunk_weights[chunk_index]
            (loss * weight).backward()
            for key in totals:
                totals[key] += float(metrics[key].cpu()) * weight
        grad_norm_total += _optimizer_step(policy, optimizer, max_grad_norm)
        optimizer_steps += 1

    return {
        **{key: value / epochs for key, value in totals.items()},
        "grad_norm": grad_norm_total / max(optimizer_steps, 1),
        "optimizer_steps": float(optimizer_steps),
        "chunks": float(len(chunks)),
    }
