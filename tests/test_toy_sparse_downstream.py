"""Tests for the noisy one-step downstream environment and GRPO bridge."""

import copy

import torch

from toy.toy_flow import D, DEVICE, VNet
from toy.toy_grpo import GRPOConfig, freeze_model, update_policy
from toy.toy_sparse_downstream import (
    NoisyPointEnv,
    collect_downstream_rollout_batch,
    evaluate_downstream_policy,
)


def tiny_config() -> GRPOConfig:
    return GRPOConfig(
        num_groups=2,
        group_size=3,
        denoising_steps=4,
        ppo_epochs=1,
        evaluation_samples=12,
    )


def make_models() -> tuple[VNet, VNet, VNet]:
    torch.manual_seed(0)
    model = VNet().to(DEVICE)
    old_model = freeze_model(copy.deepcopy(model))
    reference_model = freeze_model(copy.deepcopy(model))
    return model, old_model, reference_model


def test_zero_execution_noise_returns_commanded_action() -> None:
    environment = NoisyPointEnv(execution_noise_std=0.0)
    actions = torch.tensor([[4.0, 0.0], [0.0, 0.0]], device=DEVICE)

    executed, _ = environment.step(actions)

    torch.testing.assert_close(executed, actions)


def test_fixed_execution_noise_and_reward_use_executed_result() -> None:
    environment = NoisyPointEnv(execution_noise_std=1.0)
    actions = torch.tensor([[4.0, 0.0]], device=DEVICE)
    execution_noise = torch.tensor([[4.0, 0.0]], device=DEVICE)

    executed, rewards = environment.step(actions, execution_noise)

    torch.testing.assert_close(
        executed,
        torch.tensor([[8.0, 0.0]], device=DEVICE),
    )
    assert rewards.item() == 0.0


def test_downstream_rollout_shapes_and_detach_boundaries() -> None:
    _, old_model, reference_model = make_models()
    config = tiny_config()
    environment = NoisyPointEnv()

    batch = collect_downstream_rollout_batch(
        old_model,
        reference_model,
        config,
        environment,
    )

    assert batch["trajectory"].shape == (6, 5, D)
    assert batch["actions"].shape == (6, D)
    assert batch["executed_endpoints"].shape == (6, D)
    assert batch["old_log_probs"].shape == (6, 4)
    assert batch["reference_log_probs"].shape == (6, 4)
    assert batch["rewards"].shape == (6,)
    assert batch["advantages"].shape == (6,)
    assert all(not value.requires_grad for value in batch.values())


def test_downstream_update_changes_policy_with_mixed_advantages() -> None:
    model, old_model, reference_model = make_models()
    config = tiny_config()
    environment = NoisyPointEnv()
    batch = collect_downstream_rollout_batch(
        old_model,
        reference_model,
        config,
        environment,
    )
    batch["advantages"] = torch.tensor(
        [2 / 3, -1 / 3, -1 / 3, 1 / 3, -2 / 3, 1 / 3],
        device=DEVICE,
    )

    before = model.net[-1].bias.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = update_policy(model, optimizer, batch, config)

    assert metrics["gradient_norm"] > 0.0
    assert not torch.equal(before, model.net[-1].bias.detach())


def test_downstream_evaluation_shapes_and_reproducibility() -> None:
    model, _, _ = make_models()
    config = tiny_config()
    environment = NoisyPointEnv()
    initial_noise = torch.randn(config.evaluation_samples, D, device=DEVICE)
    execution_noise = torch.randn(
        config.evaluation_samples * 2,
        D,
        device=DEVICE,
    )

    torch.manual_seed(123)
    first_metrics, first_samples = evaluate_downstream_policy(
        model,
        config,
        environment,
        execution_repeats=2,
        initial_noise=initial_noise,
        execution_noise=execution_noise,
    )
    torch.manual_seed(123)
    second_metrics, second_samples = evaluate_downstream_policy(
        model,
        config,
        environment,
        execution_repeats=2,
        initial_noise=initial_noise,
        execution_noise=execution_noise,
    )

    assert first_metrics == second_metrics
    assert first_samples["ode_actions"].shape == (12, D)
    assert first_samples["ode_executed"].shape == (12, D)
    for name in first_samples:
        torch.testing.assert_close(first_samples[name], second_samples[name])
