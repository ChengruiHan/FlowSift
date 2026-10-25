"""SDE trajectory collection and rescoring for LeRobot SmolVLA v0.6.0.

This module deliberately wraps the upstream model without patching LeRobot.
It mirrors ``VLAFlowMatching.sample_actions`` through its public model methods,
then replaces only the Euler update with the tested SDE transition kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flowvla_rl.sde import (
    gaussian_log_prob,
    marginal_preserving_transition,
    sample_transition,
)


# 采集与打分**必须**共用的数值路径。放在这里只定义一次，是因为两边一旦漂移就会
# 静默毁掉整个 RL：同一条轨迹在 bf16 与 fp32 下算出的 log-prob 实测差 Δ=-1.07，
# 即 ratio=0.34，完全落在 clip 区间外，PPO 的 min() 恒选被 clip 的那一支，
# 梯度退化成噪声 —— 训练"在跑"、loss 有数、但成功率 75 次更新纹丝不动。
# 精度对齐后同一条轨迹逐位复现（Δ 恰好为 0，ratio 恰好为 1）。
ROLLOUT_AUTOCAST_DTYPE = torch.bfloat16


def rollout_autocast(device: torch.device | str):
    """返回采集与打分共用的 autocast 上下文。

    只在 CUDA 上启用：CPU 侧的 bf16 autocast 语义不同，而单元测试跑在 CPU 上，
    那里需要的是确定的 fp32 —— 采集本来也只在 CUDA 上发生。
    """
    dev = device if isinstance(device, torch.device) else torch.device(device)
    return torch.autocast(
        device_type=dev.type,
        dtype=ROLLOUT_AUTOCAST_DTYPE,
        enabled=(dev.type == "cuda"),
    )


@dataclass(frozen=True)
class SmolVLATrajectory:
    """One sampled action chunk and all density-bearing denoising states."""

    states: Tensor
    element_log_probs: Tensor
    valid_action_mask: Tensor
    eta: float

    @property
    def actions(self) -> Tensor:
        return self.states[:, -1]

    @property
    def log_probs(self) -> Tensor:
        """Old-policy joint log-prob for every denoising step."""
        mask = self.valid_action_mask[:, None]
        return torch.where(mask, self.element_log_probs, 0.0).sum(dim=(-1, -2))

    def masked_log_probs(self, valid_positions: Tensor) -> Tensor:
        """Old log-probs after excluding actions never executed by the env."""
        expected = (self.states.shape[0], self.states.shape[2])
        if valid_positions.shape != expected:
            raise ValueError(f"valid_positions must have shape {expected}")
        mask = self.valid_action_mask & valid_positions.to(
            device=self.states.device, dtype=torch.bool
        )[..., None]
        return torch.where(mask[:, None], self.element_log_probs, 0.0).sum(dim=(-1, -2))

    def to(self, device: torch.device | str) -> "SmolVLATrajectory":
        """Move cached tensors between rollout CPU storage and the learner GPU."""
        return SmolVLATrajectory(
            states=self.states.to(device),
            element_log_probs=self.element_log_probs.to(device),
            valid_action_mask=self.valid_action_mask.to(device),
            eta=self.eta,
        )


def make_prefix_attention_mask(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Exact local equivalent of LeRobot v0.6.0 ``make_att_2d_masks``."""
    if pad_masks.ndim != 2 or att_masks.ndim != 2:
        raise ValueError("prefix masks must both have shape (batch, sequence)")
    cumulative = torch.cumsum(att_masks, dim=1)
    causal = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = pad_masks[:, None, :] & pad_masks[:, :, None]
    return causal & valid


def build_prefix_cache(
    model: nn.Module,
    images: list[Tensor],
    image_masks: list[Tensor],
    language_tokens: Tensor,
    language_masks: Tensor,
    state: Tensor,
) -> tuple[Tensor, object]:
    """Build the frozen observation cache shared by all denoising steps."""
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, image_masks, language_tokens, language_masks, state=state
    )
    attention = make_prefix_attention_mask(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    _, cache = model.vlm_with_expert.forward(
        attention_mask=attention,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=model.config.use_cache,
        fill_kv_cache=True,
    )
    return prefix_pad_masks, cache


def prepare_policy_prefix(policy: nn.Module, batch: dict[str, Tensor]) -> tuple[Tensor, object]:
    """Convert an already normalized LeRobot batch into SmolVLA's prefix cache."""
    prepared = dict(batch)
    prepared = policy._prepare_batch(prepared)
    images, image_masks = policy.prepare_images(prepared)
    state = policy.prepare_state(prepared)
    try:
        language_tokens = prepared["observation.language.tokens"]
        language_masks = prepared["observation.language.attention_mask"]
    except KeyError as exc:
        raise KeyError(
            "batch must be passed through the SmolVLA policy preprocessor first"
        ) from exc
    return build_prefix_cache(
        policy.model,
        images,
        image_masks,
        language_tokens,
        language_masks,
        state,
    )


def valid_action_mask(
    batch_size: int,
    chunk_size: int,
    max_action_dim: int,
    action_dim: int,
    *,
    device: torch.device | str,
    valid_positions: Tensor | None = None,
) -> Tensor:
    """Mask padded dimensions and, after rollout, unexecuted chunk positions."""
    if not 0 < action_dim <= max_action_dim:
        raise ValueError("action_dim must lie in (0, max_action_dim]")
    mask = torch.arange(max_action_dim, device=device)[None, None, :] < action_dim
    mask = mask.expand(batch_size, chunk_size, -1)
    if valid_positions is not None:
        if valid_positions.shape != (batch_size, chunk_size):
            raise ValueError(
                f"valid_positions must have shape {(batch_size, chunk_size)}"
            )
        mask = mask & valid_positions.to(device=device, dtype=torch.bool)[..., None]
    return mask


def sample_sde_chunk(
    model: nn.Module,
    prefix_pad_masks: Tensor,
    prefix_cache: object,
    *,
    action_dim: int,
    eta: float,
    noise: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> SmolVLATrajectory:
    """Sample one SmolVLA chunk and cache old-policy log-probs.

    RTC is intentionally excluded: changing denoising updates behind the RTC
    processor would make the likelihood model inconsistent with sampling.
    """
    if eta <= 0:
        raise ValueError("eta must be positive for a density-bearing rollout")
    if getattr(model, "_rtc_enabled", lambda: False)():
        raise NotImplementedError("RTC is not supported by the RL SDE sampler")
    batch_size = prefix_pad_masks.shape[0]
    shape = (batch_size, model.config.chunk_size, model.config.max_action_dim)
    if noise is None:
        noise = model.sample_noise(shape, prefix_pad_masks.device)
    elif noise.shape != shape:
        raise ValueError(f"noise must have shape {shape}, got {tuple(noise.shape)}")

    mask = valid_action_mask(
        batch_size,
        model.config.chunk_size,
        model.config.max_action_dim,
        action_dim,
        device=noise.device,
    )
    step_size = 1.0 / model.config.num_steps
    state = noise.detach().float()
    states = [state]
    element_log_probs = []
    with torch.no_grad():
        for step in range(model.config.num_steps):
            time = 1.0 - step * step_size
            time_tensor = torch.full(
                (batch_size,), time, dtype=torch.float32, device=state.device
            )
            velocity = model.denoise_step(
                x_t=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=prefix_cache,
                timestep=time_tensor,
            )
            transition = marginal_preserving_transition(
                state, velocity, time, step_size, eta
            )
            next_state = sample_transition(transition, generator=generator)
            element_log_probs.append(
                gaussian_log_prob(
                    next_state,
                    transition.mean,
                    transition.std,
                    mask=mask,
                    reduction="none",
                )
            )
            states.append(next_state)
            state = next_state
    return SmolVLATrajectory(
        states=torch.stack(states, dim=1).detach(),
        element_log_probs=torch.stack(element_log_probs, dim=1).detach(),
        valid_action_mask=mask.detach(),
        eta=float(eta),
    )


def recompute_log_probs(
    model: nn.Module,
    prefix_pad_masks: Tensor,
    prefix_cache: object,
    trajectory: SmolVLATrajectory,
    *,
    valid_positions: Tensor | None = None,
) -> Tensor:
    """Rescore fixed sampled states under current or reference expert weights."""
    states = trajectory.states.detach()
    batch_size, state_count, chunk_size, max_action_dim = states.shape
    if state_count != model.config.num_steps + 1:
        raise ValueError("trajectory state count does not match model num_steps")
    if (chunk_size, max_action_dim) != (
        model.config.chunk_size,
        model.config.max_action_dim,
    ):
        raise ValueError("trajectory action shape does not match model configuration")
    mask = trajectory.valid_action_mask
    if valid_positions is not None:
        mask = mask & valid_positions.to(device=states.device, dtype=torch.bool)[..., None]

    step_size = 1.0 / model.config.num_steps
    log_probs = []
    for step in range(model.config.num_steps):
        time = 1.0 - step * step_size
        time_tensor = torch.full(
            (batch_size,), time, dtype=torch.float32, device=states.device
        )
        velocity = model.denoise_step(
            x_t=states[:, step],
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=prefix_cache,
            timestep=time_tensor,
        )
        transition = marginal_preserving_transition(
            states[:, step], velocity, time, step_size, trajectory.eta
        )
        log_probs.append(
            gaussian_log_prob(
                states[:, step + 1], transition.mean, transition.std, mask=mask
            )
        )
    return torch.stack(log_probs, dim=1)
