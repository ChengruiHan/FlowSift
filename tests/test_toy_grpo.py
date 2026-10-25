"""Formula-level tests for the Day 3 toy GRPO implementation."""

import copy

import torch

from toy.toy_flow import D, DEVICE, VNet
from toy.toy_grpo import (
    GRPOConfig,
    collect_rollout_batch,
    compute_group_advantages,
    compute_group_diagnostics,
    compute_grpo_loss,
    freeze_model,
    k3_kl_penalty,
    terminal_reward,
    update_policy,
)
from toy.toy_sde import sde_rollout, trajectory_log_probs


def make_model() -> VNet:
    torch.manual_seed(0)
    return VNet().to(DEVICE)


def tiny_config(**overrides: object) -> GRPOConfig:
    values: dict[str, object] = {
        "num_groups": 2,
        "group_size": 3,
        "denoising_steps": 4,
        "eta": 0.5,
        "ppo_epochs": 1,
        "evaluation_samples": 16,
    }
    values.update(overrides)
    return GRPOConfig(**values)


def test_terminal_reward() -> None:
    endpoints = torch.tensor(
        [[4.0, 0.0], [4.5, 0.0], [0.0, 0.0]],
        device=DEVICE,
    )
    center = torch.tensor([4.0, 0.0], device=DEVICE)

    rewards = terminal_reward(endpoints, center, radius=1.0)

    torch.testing.assert_close(
        rewards,
        torch.tensor([1.0, 1.0, 0.0], device=DEVICE),
    )


def test_group_advantages_match_hand_calculation() -> None:
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    advantages = compute_group_advantages(
        rewards,
        num_groups=2,
        group_size=3,
    )
    expected = torch.tensor(
        [2 / 3, -1 / 3, -1 / 3, 1 / 3, -2 / 3, 1 / 3]
    )

    torch.testing.assert_close(advantages, expected)
    grouped = advantages.reshape(2, 3)
    torch.testing.assert_close(grouped.sum(dim=1), torch.zeros(2))
    assert not advantages.requires_grad


def test_group_diagnostics_identify_mixed_and_degenerate_groups() -> None:
    rewards = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    diagnostics = compute_group_diagnostics(
        rewards,
        num_groups=2,
        group_size=3,
    )

    assert diagnostics["mixed_group_fraction"].item() == 0.5
    assert diagnostics["all_failure_group_fraction"].item() == 0.0
    assert diagnostics["all_success_group_fraction"].item() == 0.5
    assert diagnostics["zero_advantage_group_fraction"].item() == 0.5


def test_rollout_batch_shapes_and_detach_boundaries() -> None:
    model = make_model()
    old_model = freeze_model(copy.deepcopy(model))
    reference_model = freeze_model(copy.deepcopy(model))
    config = tiny_config()
    center = torch.tensor([4.0, 0.0], device=DEVICE)

    batch = collect_rollout_batch(
        old_model,
        reference_model,
        config,
        center,
    )

    assert batch["trajectory"].shape == (6, 5, D)
    assert batch["old_log_probs"].shape == (6, 4)
    assert batch["reference_log_probs"].shape == (6, 4)
    assert batch["rewards"].shape == (6,)
    assert batch["advantages"].shape == (6,)
    assert all(not tensor.requires_grad for tensor in batch.values())

    # Independent policy samples within a group should not repeat x_0.
    assert not torch.equal(
        batch["trajectory"][0, 0],
        batch["trajectory"][1, 0],
    )


def test_k3_uses_reference_over_current_ratio() -> None:
    log_two = torch.log(torch.tensor(2.0))
    new_log_probs = torch.tensor([[log_two]])
    reference_log_probs = torch.zeros_like(new_log_probs)

    actual = k3_kl_penalty(new_log_probs, reference_log_probs)
    # r = ref/current = 1/2, so k3 = 1/2 - 1 + log(2).
    expected = torch.tensor([[0.5 - 1.0 + log_two.item()]])

    torch.testing.assert_close(actual, expected)
    assert actual.item() >= 0.0
    torch.testing.assert_close(
        k3_kl_penalty(reference_log_probs, reference_log_probs),
        torch.zeros_like(reference_log_probs),
    )


def make_mixed_reward_batch(
    model: VNet,
    config: GRPOConfig,
) -> dict[str, torch.Tensor]:
    trajectory = sde_rollout(
        model,
        n=config.num_groups * config.group_size,
        steps=config.denoising_steps,
        eta=config.eta,
    )
    with torch.no_grad():
        old_log_probs = trajectory_log_probs(model, trajectory, config.eta)

    rewards = torch.tensor(
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        device=DEVICE,
    )
    advantages = compute_group_advantages(
        rewards,
        config.num_groups,
        config.group_size,
    )
    return {
        "trajectory": trajectory.detach(),
        "old_log_probs": old_log_probs.detach(),
        "reference_log_probs": old_log_probs.detach().clone(),
        "rewards": rewards.detach(),
        "advantages": advantages.detach(),
    }


def test_initial_ratio_and_kl_are_one_and_zero() -> None:
    model = make_model()
    config = tiny_config()
    batch = make_mixed_reward_batch(model, config)

    loss, metrics = compute_grpo_loss(model, batch, config)

    assert loss.requires_grad
    torch.testing.assert_close(metrics["mean_ratio"], metrics["mean_ratio"].new_tensor(1.0))
    torch.testing.assert_close(metrics["kl_loss"], metrics["kl_loss"].new_tensor(0.0))


def test_mixed_advantages_produce_gradient_and_policy_update() -> None:
    model = make_model()
    config = tiny_config(learning_rate=1e-3)
    batch = make_mixed_reward_batch(model, config)

    loss, _ = compute_grpo_loss(model, batch, config)
    loss.backward()
    gradient = model.net[-1].bias.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.norm().item() > 0.0

    before = model.net[-1].bias.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metrics = update_policy(model, optimizer, batch, config)
    after = model.net[-1].bias.detach()

    assert not torch.equal(before, after)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
