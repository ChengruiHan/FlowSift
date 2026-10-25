"""Critic-free GRPO objectives for denoising trajectories."""

from __future__ import annotations

import torch
from torch import Tensor


def compute_group_advantages(
    rewards: Tensor,
    *,
    use_std: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Center complete-episode rewards within each reset-matched group.

    Input shape is ``(number_of_groups, group_size)``.  The default deliberately
    does not divide by standard deviation (the project's Dr.GRPO convention).
    """
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape (groups, group_size>=2)")
    advantages = rewards.float() - rewards.float().mean(dim=1, keepdim=True)
    if use_std:
        advantages = advantages / (rewards.float().std(dim=1, keepdim=True, unbiased=False) + eps)
    return advantages.detach()


def k3_kl_estimate(logp: Tensor, ref_logp: Tensor) -> Tensor:
    """Non-negative k3 estimate of KL(policy || reference).

    With ``log_ratio = log p_ref - log p_policy``, k3 is
    ``exp(log_ratio) - log_ratio - 1``.
    """
    log_ratio = ref_logp.detach().float() - logp.float()
    return torch.exp(log_ratio) - log_ratio - 1.0


def grpo_loss(
    logp: Tensor,
    old_logp: Tensor,
    ref_logp: Tensor,
    advantages: Tensor,
    *,
    clip_epsilon: float,
    kl_beta: float,
    valid_steps: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute per-denoise-step clipped GRPO plus in-loss reference KL.

    ``logp`` has shape ``(episodes, control_chunks, denoise_steps)``. One scalar
    episode advantage is broadcast across both inner axes. There is no
    transition probability between adjacent control chunks.
    """
    if logp.ndim != 3:
        raise ValueError("logp must have shape (episodes, chunks, denoise_steps)")
    if old_logp.shape != logp.shape or ref_logp.shape != logp.shape:
        raise ValueError("old_logp and ref_logp must match logp")
    if advantages.shape != (logp.shape[0],):
        raise ValueError("advantages must contain one scalar per episode")
    if clip_epsilon < 0 or kl_beta < 0:
        raise ValueError("clip_epsilon and kl_beta must be non-negative")

    if valid_steps is None:
        valid = torch.ones_like(logp, dtype=torch.bool)
    else:
        try:
            valid = torch.broadcast_to(valid_steps.to(device=logp.device, dtype=torch.bool), logp.shape)
        except RuntimeError as exc:
            raise ValueError("valid_steps is not broadcastable to logp") from exc
    valid_f = valid.float()
    denominator = valid_f.sum().clamp_min(1.0)

    log_ratio = (logp.float() - old_logp.detach().float()).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    advantage = advantages.detach().float()[:, None, None]
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    pg_loss = -(torch.minimum(unclipped, clipped) * valid_f).sum() / denominator

    kl = k3_kl_estimate(logp, ref_logp)
    kl_loss = (kl * valid_f).sum() / denominator
    loss = pg_loss + float(kl_beta) * kl_loss
    metrics = {
        "loss": loss.detach(),
        "pg_loss": pg_loss.detach(),
        "kl": kl_loss.detach(),
        "ratio_mean": (ratio.detach() * valid_f).sum() / denominator,
        "clip_fraction": (((ratio - 1.0).abs() > clip_epsilon).float() * valid_f).sum()
        / denominator,
        "advantage_mean": advantages.detach().float().mean(),
        "advantage_std": advantages.detach().float().std(unbiased=False),
    }
    return loss, metrics

