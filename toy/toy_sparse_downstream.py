"""Day 3.3: GRPO when actions receive reward only after noisy execution."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

if __package__:
    from toy.toy_flow import D, DEVICE, VNet, ode_sample, train_cfm
    from toy.toy_grpo import (
        GRPOConfig,
        compute_group_advantages,
        compute_group_diagnostics,
        freeze_model,
        terminal_reward,
        update_policy,
    )
    from toy.toy_sde import sde_rollout, trajectory_log_probs
    from toy.visualization import save_sample_comparison
else:
    from toy_flow import D, DEVICE, VNet, ode_sample, train_cfm
    from toy_grpo import (
        GRPOConfig,
        compute_group_advantages,
        compute_group_diagnostics,
        freeze_model,
        terminal_reward,
        update_policy,
    )
    from toy_sde import sde_rollout, trajectory_log_probs
    from visualization import save_sample_comparison


BASE_CHECKPOINT_PATH = Path("results/day3_toy_grpo_checkpoint.pt")
ENDPOINT_FIGURE_PATH = Path("results/day3_sparse_downstream_endpoints.png")
TRAINING_FIGURE_PATH = Path("results/day3_sparse_downstream_training.png")
CHECKPOINT_PATH = Path("results/day3_sparse_downstream_checkpoint.pt")


@dataclass(frozen=True)
class NoisyPointEnv:
    """One-step environment: execute a commanded point with Gaussian error."""

    target_center_x: float = 4.0
    target_center_y: float = 0.0
    reward_radius: float = 1.0
    execution_noise_std: float = 0.5

    def target_center(self, reference: torch.Tensor) -> torch.Tensor:
        return reference.new_tensor(
            [self.target_center_x, self.target_center_y]
        )

    @torch.no_grad()
    def step(
        self,
        actions: torch.Tensor,
        execution_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return executed endpoints and binary terminal rewards."""
        if actions.ndim != 2 or actions.shape[-1] != D:
            raise ValueError(
                f"actions must have shape (batch, {D}), "
                f"got {tuple(actions.shape)}"
            )
        if self.execution_noise_std < 0:
            raise ValueError("execution_noise_std must be non-negative")

        if execution_noise is None:
            execution_noise = torch.randn_like(actions)
        elif execution_noise.shape != actions.shape:
            raise ValueError("execution_noise must have the same shape as actions")

        executed_endpoints = (
            actions + self.execution_noise_std * execution_noise
        )
        rewards = terminal_reward(
            endpoints=executed_endpoints,
            target_center=self.target_center(actions),
            radius=self.reward_radius,
        )
        return executed_endpoints, rewards


@torch.no_grad()
def collect_downstream_rollout_batch(
    old_model: nn.Module,
    reference_model: nn.Module,
    config: GRPOConfig,
    environment: NoisyPointEnv,
) -> dict[str, torch.Tensor]:
    """Collect policy trajectories, then obtain reward from the environment."""
    batch_size = config.num_groups * config.group_size
    device = next(old_model.parameters()).device
    initial_noise = torch.randn(batch_size, D, device=device)

    trajectory = sde_rollout(
        model=old_model,
        n=batch_size,
        steps=config.denoising_steps,
        eta=config.eta,
        initial_noise=initial_noise,
    )
    old_log_probs = trajectory_log_probs(
        model=old_model,
        trajectory=trajectory,
        eta=config.eta,
    )
    reference_log_probs = trajectory_log_probs(
        model=reference_model,
        trajectory=trajectory,
        eta=config.eta,
    )

    actions = trajectory[:, -1]
    executed_endpoints, rewards = environment.step(actions)
    advantages = compute_group_advantages(
        rewards=rewards,
        num_groups=config.num_groups,
        group_size=config.group_size,
        use_std=config.use_advantage_std,
    )
    diagnostics = compute_group_diagnostics(
        rewards=rewards,
        num_groups=config.num_groups,
        group_size=config.group_size,
    )

    return {
        "trajectory": trajectory.detach(),
        "actions": actions.detach(),
        "executed_endpoints": executed_endpoints.detach(),
        "old_log_probs": old_log_probs.detach(),
        "reference_log_probs": reference_log_probs.detach(),
        "rewards": rewards.detach(),
        "advantages": advantages.detach(),
        **diagnostics,
    }


@torch.no_grad()
def evaluate_downstream_policy(
    model: nn.Module,
    config: GRPOConfig,
    environment: NoisyPointEnv,
    execution_repeats: int = 4,
    initial_noise: torch.Tensor | None = None,
    execution_noise: torch.Tensor | None = None,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Evaluate commanded and executed success for both ODE and SDE actions."""
    if execution_repeats <= 0:
        raise ValueError("execution_repeats must be positive")

    model.eval()
    device = next(model.parameters()).device
    sample_count = config.evaluation_samples
    if initial_noise is None:
        initial_noise = torch.randn(sample_count, D, device=device)
    elif initial_noise.shape != (sample_count, D):
        raise ValueError(
            f"initial_noise must have shape {(sample_count, D)}, "
            f"got {tuple(initial_noise.shape)}"
        )
    else:
        initial_noise = initial_noise.to(device)

    ode_actions = ode_sample(
        model=model,
        n=sample_count,
        steps=config.denoising_steps,
        initial_noise=initial_noise,
    )
    sde_actions = sde_rollout(
        model=model,
        n=sample_count,
        steps=config.denoising_steps,
        eta=config.eta,
        initial_noise=initial_noise,
    )[:, -1]

    target_center = environment.target_center(ode_actions)
    ode_command_rewards = terminal_reward(
        ode_actions,
        target_center,
        environment.reward_radius,
    )
    sde_command_rewards = terminal_reward(
        sde_actions,
        target_center,
        environment.reward_radius,
    )

    repeated_ode_actions = ode_actions.repeat_interleave(execution_repeats, dim=0)
    repeated_sde_actions = sde_actions.repeat_interleave(execution_repeats, dim=0)
    repeated_count = sample_count * execution_repeats
    if execution_noise is None:
        execution_noise = torch.randn(repeated_count, D, device=device)
    elif execution_noise.shape != (repeated_count, D):
        raise ValueError(
            f"execution_noise must have shape {(repeated_count, D)}, "
            f"got {tuple(execution_noise.shape)}"
        )
    else:
        execution_noise = execution_noise.to(device)

    ode_executed, ode_executed_rewards = environment.step(
        repeated_ode_actions,
        execution_noise=execution_noise,
    )
    sde_executed, sde_executed_rewards = environment.step(
        repeated_sde_actions,
        execution_noise=execution_noise,
    )

    metrics = {
        "ode_command_success_rate": ode_command_rewards.mean().item(),
        "sde_command_success_rate": sde_command_rewards.mean().item(),
        "ode_executed_success_rate": ode_executed_rewards.mean().item(),
        "sde_executed_success_rate": sde_executed_rewards.mean().item(),
    }
    samples = {
        "ode_actions": ode_actions,
        "sde_actions": sde_actions,
        "ode_executed": ode_executed.reshape(
            sample_count, execution_repeats, D
        )[:, 0],
        "sde_executed": sde_executed.reshape(
            sample_count, execution_repeats, D
        )[:, 0],
    }
    return metrics, samples


def train_downstream_grpo(
    model: nn.Module,
    reference_model: nn.Module,
    config: GRPOConfig,
    environment: NoisyPointEnv,
    execution_repeats: int = 4,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Train with rewards produced only after noisy environment execution."""
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    metric_names = (
        "total_loss",
        "policy_loss",
        "kl_loss",
        "mean_ratio",
        "clip_fraction",
        "approximate_old_kl",
        "gradient_norm",
    )
    diagnostic_names = (
        "mixed_group_fraction",
        "all_failure_group_fraction",
        "all_success_group_fraction",
        "zero_advantage_group_fraction",
    )
    evaluation_names = (
        "ode_command_success_rate",
        "sde_command_success_rate",
        "ode_executed_success_rate",
        "sde_executed_success_rate",
    )
    history = {
        name: []
        for name in (
            "iteration",
            "rollout_reward",
            "advantage_abs_mean",
            *metric_names,
            *diagnostic_names,
            *evaluation_names,
        )
    }

    initial_evaluation, _ = evaluate_downstream_policy(
        model,
        config,
        environment,
        execution_repeats,
    )
    latest_evaluation = initial_evaluation
    print(
        "\nInitial downstream policy: "
        f"ODE command={initial_evaluation['ode_command_success_rate']:.4f}, "
        f"ODE executed={initial_evaluation['ode_executed_success_rate']:.4f}, "
        f"SDE executed={initial_evaluation['sde_executed_success_rate']:.4f}"
    )

    for iteration in range(1, config.training_iterations + 1):
        old_model = freeze_model(copy.deepcopy(model))
        rollout_batch = collect_downstream_rollout_batch(
            old_model,
            reference_model,
            config,
            environment,
        )
        update_metrics = update_policy(
            model,
            optimizer,
            rollout_batch,
            config,
        )

        if (
            iteration == 1
            or iteration % config.evaluation_interval == 0
            or iteration == config.training_iterations
        ):
            latest_evaluation, _ = evaluate_downstream_policy(
                model,
                config,
                environment,
                execution_repeats,
            )

        history["iteration"].append(float(iteration))
        history["rollout_reward"].append(
            rollout_batch["rewards"].mean().item()
        )
        history["advantage_abs_mean"].append(
            rollout_batch["advantages"].abs().mean().item()
        )
        for name in metric_names:
            history[name].append(update_metrics[name])
        for name in diagnostic_names:
            history[name].append(rollout_batch[name].item())
        for name in evaluation_names:
            history[name].append(latest_evaluation[name])

        if iteration == 1 or iteration % config.evaluation_interval == 0:
            print(
                f"Iteration {iteration:4d} | "
                f"reward={history['rollout_reward'][-1]:.4f} | "
                f"mixed={history['mixed_group_fraction'][-1]:.4f} | "
                f"KL={update_metrics['kl_loss']:.4f} | "
                f"ratio={update_metrics['mean_ratio']:.4f} | "
                f"grad={update_metrics['gradient_norm']:.4f} | "
                f"ODE cmd={latest_evaluation['ode_command_success_rate']:.4f} | "
                f"ODE exec={latest_evaluation['ode_executed_success_rate']:.4f} | "
                f"SDE exec={latest_evaluation['sde_executed_success_rate']:.4f}"
            )

    return history, initial_evaluation


def save_downstream_training_curves(
    history: dict[str, list[float]],
    output_path: str | Path,
) -> Path:
    """Save task, robustness, KL, group, and optimization diagnostics."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    iterations = history["iteration"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    axes[0, 0].plot(iterations, history["rollout_reward"], label="Rollout")
    axes[0, 0].plot(
        iterations, history["ode_executed_success_rate"], label="ODE executed"
    )
    axes[0, 0].plot(
        iterations, history["sde_executed_success_rate"], label="SDE executed"
    )
    axes[0, 0].set_title("Downstream reward")

    axes[0, 1].plot(
        iterations, history["ode_command_success_rate"], label="ODE command"
    )
    axes[0, 1].plot(
        iterations, history["ode_executed_success_rate"], label="ODE executed"
    )
    axes[0, 1].set_title("Command-to-execution gap")

    axes[0, 2].plot(iterations, history["kl_loss"], label="Reference k3 KL")
    axes[0, 2].plot(
        iterations, history["approximate_old_kl"], label="Old-policy KL"
    )
    axes[0, 2].set_title("KL diagnostics")

    axes[1, 0].plot(
        iterations, history["mixed_group_fraction"], label="Mixed groups"
    )
    axes[1, 0].plot(
        iterations,
        history["zero_advantage_group_fraction"],
        label="Zero-advantage groups",
    )
    axes[1, 0].set_title("Group diagnostics")

    axes[1, 1].plot(iterations, history["gradient_norm"], label="Gradient norm")
    axes[1, 1].plot(
        iterations, history["advantage_abs_mean"], label="Mean |advantage|"
    )
    axes[1, 1].set_title("Learning signal")

    axes[1, 2].plot(iterations, history["mean_ratio"], label="Mean ratio")
    axes[1, 2].plot(iterations, history["clip_fraction"], label="Clip fraction")
    axes[1, 2].set_title("PPO diagnostics")

    for ax in axes.flat:
        ax.set_xlabel("GRPO iteration")
        ax.grid(alpha=0.2)
        ax.legend()

    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def load_or_train_reference_state() -> dict[str, torch.Tensor]:
    """Use the pre-GRPO CFM reference, with a reproducible fallback."""
    if BASE_CHECKPOINT_PATH.exists():
        checkpoint = torch.load(
            BASE_CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
        return checkpoint["reference_state_dict"]

    model = VNet().to(DEVICE)
    train_cfm(model, steps=5000, batch_size=512, lr=1e-3)
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def reward_circle(environment: NoisyPointEnv, reference: torch.Tensor) -> torch.Tensor:
    angles = torch.linspace(0.0, 2.0 * math.pi, 256, device=reference.device)
    return environment.target_center(reference)[None] + environment.reward_radius * torch.stack(
        [torch.cos(angles), torch.sin(angles)], dim=1
    )


def main() -> None:
    torch.manual_seed(0)
    print(f"Using device: {DEVICE}")

    reference_state = load_or_train_reference_state()
    model = VNet().to(DEVICE)
    model.load_state_dict(reference_state)
    reference_model = freeze_model(copy.deepcopy(model))
    config = GRPOConfig()
    environment = NoisyPointEnv(
        target_center_x=config.target_center_x,
        target_center_y=config.target_center_y,
        reward_radius=config.reward_radius,
        execution_noise_std=0.5,
    )

    comparison_count = 5000
    comparison_noise = torch.randn(comparison_count, D, device=DEVICE)
    comparison_execution_noise = torch.randn_like(comparison_noise)
    before_actions = ode_sample(
        model,
        n=comparison_count,
        steps=config.denoising_steps,
        initial_noise=comparison_noise,
    )
    before_executed, _ = environment.step(
        before_actions,
        execution_noise=comparison_execution_noise,
    )

    history, initial_evaluation = train_downstream_grpo(
        model,
        reference_model,
        config,
        environment,
        execution_repeats=4,
    )

    after_actions = ode_sample(
        model,
        n=comparison_count,
        steps=config.denoising_steps,
        initial_noise=comparison_noise,
    )
    after_executed, _ = environment.step(
        after_actions,
        execution_noise=comparison_execution_noise,
    )
    final_evaluation, _ = evaluate_downstream_policy(
        model,
        config,
        environment,
        execution_repeats=4,
    )

    endpoint_path = save_sample_comparison(
        panels={
            "Before: command": before_actions,
            "Before: executed": before_executed,
            "After: command": after_actions,
            "After: executed": after_executed,
            "Reward region": reward_circle(environment, after_actions),
        },
        output_path=ENDPOINT_FIGURE_PATH,
    )
    training_path = save_downstream_training_curves(
        history,
        TRAINING_FIGURE_PATH,
    )
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "reference_state_dict": reference_model.state_dict(),
            "grpo_config": asdict(config),
            "environment_config": asdict(environment),
            "initial_evaluation": initial_evaluation,
            "final_evaluation": final_evaluation,
            "history": history,
        },
        CHECKPOINT_PATH,
    )

    print("\nInitial evaluation:", initial_evaluation)
    print("Final evaluation:", final_evaluation)
    print(f"Saved endpoint comparison to: {endpoint_path}")
    print(f"Saved training curves to: {training_path}")
    print(f"Saved checkpoint to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
