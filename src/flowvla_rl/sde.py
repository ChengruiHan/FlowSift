"""Marginal-preserving stochastic sampling for linear flow matching.

The sampler follows SmolVLA's time convention: ``t=1`` is Gaussian noise,
``t=0`` is an action, and one denoising step moves from ``t`` to ``t-h``.
All probability calculations are performed in float32 even when the velocity
model runs in bf16.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Transition:
    """Parameters of one reverse-time Gaussian transition."""

    mean: Tensor
    std: Tensor
    score: Tensor
    reverse_drift: Tensor


def _broadcast_time(time: Tensor | float, reference: Tensor) -> Tensor:
    value = torch.as_tensor(time, device=reference.device, dtype=torch.float32)
    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    if value.shape != (reference.shape[0],):
        raise ValueError(
            f"time must be scalar or shape {(reference.shape[0],)}, got {tuple(value.shape)}"
        )
    return value.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


def score_from_velocity(x_t: Tensor, velocity: Tensor, time: Tensor | float) -> Tensor:
    """Recover the linear-path score from a flow-matching velocity.

    For ``x_t = t * noise + (1-t) * action`` and
    ``v = noise - action``, the score is
    ``-(x_t + (1-t) * v) / t``.
    """
    if x_t.shape != velocity.shape:
        raise ValueError("x_t and velocity must have identical shapes")
    time_b = _broadcast_time(time, x_t)
    if torch.any(time_b <= 0) or torch.any(time_b > 1):
        raise ValueError("time must lie in (0, 1]")
    x_f = x_t.float()
    velocity_f = velocity.float()
    return -(x_f + (1.0 - time_b) * velocity_f) / time_b


def diffusion_scale(time: Tensor | float, eta: float, reference: Tensor) -> Tensor:
    """Return the bounded schedule ``g(t)=eta*sqrt(t)`` in float32."""
    if eta < 0:
        raise ValueError("eta must be non-negative")
    time_b = _broadcast_time(time, reference)
    if torch.any(time_b <= 0) or torch.any(time_b > 1):
        raise ValueError("time must lie in (0, 1]")
    return float(eta) * torch.sqrt(time_b)


def marginal_preserving_transition(
    x_t: Tensor,
    velocity: Tensor,
    time: Tensor | float,
    step_size: float,
    eta: float,
) -> Transition:
    """Build ``p(x_{t-h}|x_t)`` for the marginal-preserving reverse SDE.

    ``mean = x_t - h * (v - 0.5*g(t)^2*score)`` and
    ``std = g(t)*sqrt(h)``.  The returned tensors are float32 to prevent
    overflow in Gaussian likelihoods.
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if x_t.shape != velocity.shape:
        raise ValueError("x_t and velocity must have identical shapes")
    score = score_from_velocity(x_t, velocity, time)
    g_t = diffusion_scale(time, eta, x_t)
    reverse_drift = velocity.float() - 0.5 * g_t.square() * score
    mean = x_t.float() - float(step_size) * reverse_drift
    std = g_t * math.sqrt(step_size)
    return Transition(mean=mean, std=std, score=score, reverse_drift=reverse_drift)


def sample_transition(
    transition: Transition,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Draw one sample without retaining a graph through the sampled state."""
    if torch.any(transition.std <= 0):
        raise ValueError("sampling with log-prob requires eta > 0")
    noise = torch.randn(
        transition.mean.shape,
        dtype=transition.mean.dtype,
        device=transition.mean.device,
        generator=generator,
    )
    return (transition.mean + transition.std * noise).detach()


def gaussian_log_prob(
    sample: Tensor,
    mean: Tensor,
    std: Tensor,
    *,
    mask: Tensor | None = None,
    reduction: str = "sum",
) -> Tensor:
    """Evaluate a diagonal Gaussian and reduce non-batch dimensions.

    ``mask`` is broadcast to ``sample`` and is essential for SmolVLA: padded
    action dimensions and padded action positions must not enter the policy
    ratio. ``reduction='mean'`` divides by the number of valid elements and is
    available for diagnostics; the frozen project convention uses ``sum``.
    """
    if sample.shape != mean.shape:
        raise ValueError("sample and mean must have identical shapes")
    if reduction not in {"sum", "mean", "none"}:
        raise ValueError("reduction must be one of: sum, mean, none")

    sample_f, mean_f, std_f = sample.float(), mean.float(), std.float()
    try:
        std_f = torch.broadcast_to(std_f, sample_f.shape)
    except RuntimeError as exc:
        raise ValueError("std is not broadcastable to sample") from exc
    if torch.any(std_f <= 0):
        raise ValueError("std must be strictly positive")

    elementwise = -0.5 * (
        ((sample_f.detach() - mean_f) / std_f).square()
        + 2.0 * torch.log(std_f)
        + math.log(2.0 * math.pi)
    )
    if mask is None:
        valid = torch.ones_like(elementwise, dtype=torch.bool)
    else:
        try:
            valid = torch.broadcast_to(mask.to(device=sample.device, dtype=torch.bool), sample.shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to sample") from exc
    elementwise = torch.where(valid, elementwise, torch.zeros_like(elementwise))
    if reduction == "none":
        return elementwise

    reduce_dims = tuple(range(1, elementwise.ndim))
    total = elementwise.sum(dim=reduce_dims)
    if reduction == "sum":
        return total
    count = valid.sum(dim=reduce_dims).clamp_min(1)
    return total / count

