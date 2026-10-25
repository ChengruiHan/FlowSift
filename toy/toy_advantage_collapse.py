"""Day 3.2: reproduce GRPO advantage collapse and recover by increasing G."""

from __future__ import annotations

import copy
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import torch

if __package__:
    from toy.toy_flow import DEVICE, VNet, train_cfm
    from toy.toy_grpo import GRPOConfig, freeze_model, terminal_reward, train_grpo
    from toy.toy_sde import sde_rollout
else:
    from toy_flow import DEVICE, VNet, train_cfm
    from toy_grpo import GRPOConfig, freeze_model, terminal_reward, train_grpo
    from toy_sde import sde_rollout


CHECKPOINT_PATH = Path("results/day3_toy_grpo_checkpoint.pt")
FIGURE_PATH = Path("results/day3_advantage_collapse_vs_recovery.png")
HISTORY_PATH = Path("results/day3_advantage_collapse_history.pt")
RADIUS_CANDIDATES = (0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2)


def load_or_train_reference_state() -> dict[str, torch.Tensor]:
    """Load the pre-GRPO CFM state, or recreate it when results are absent."""
    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
        return checkpoint["reference_state_dict"]

    print("Day 3 checkpoint not found; training a fresh CFM reference...")
    model = VNet().to(DEVICE)
    train_cfm(model, steps=5000, batch_size=512, lr=1e-3)
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


@torch.no_grad()
def calibrate_reward_radius(
    reference_state: dict[str, torch.Tensor],
    target_success_rate: float = 0.01,
    sample_count: int = 20000,
    seed: int = 123,
) -> tuple[float, dict[float, float]]:
    """Choose the candidate reward radius closest to the target rarity."""
    torch.manual_seed(seed)
    model = VNet().to(DEVICE)
    model.load_state_dict(reference_state)
    model.eval()

    endpoints = sde_rollout(
        model=model,
        n=sample_count,
        steps=20,
        eta=0.5,
    )[:, -1]
    target_center = torch.tensor([4.0, 0.0], device=DEVICE)

    success_rates = {
        radius: terminal_reward(
            endpoints=endpoints,
            target_center=target_center,
            radius=radius,
        ).mean().item()
        for radius in RADIUS_CANDIDATES
    }
    selected_radius = min(
        success_rates,
        key=lambda radius: abs(success_rates[radius] - target_success_rate),
    )
    return selected_radius, success_rates


def mixed_group_probability(success_rate: float, group_size: int) -> float:
    """P(mixed) = 1 - p^G - (1-p)^G for binary rewards."""
    return 1.0 - success_rate**group_size - (1.0 - success_rate) ** group_size


def run_condition(
    reference_state: dict[str, torch.Tensor],
    config: GRPOConfig,
    seed: int,
) -> dict[str, list[float]]:
    """Start from the same CFM policy and run one controlled condition."""
    torch.manual_seed(seed)
    model = VNet().to(DEVICE)
    model.load_state_dict(reference_state)
    reference_model = freeze_model(copy.deepcopy(model))
    return train_grpo(
        model=model,
        reference_model=reference_model,
        config=config,
    )


def save_comparison_figure(
    collapse_history: dict[str, list[float]],
    recovery_history: dict[str, list[float]],
    output_path: str | Path,
) -> Path:
    """Save the causal chain from degenerate groups to zero gradients."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = (
        ("rollout_reward", "Rollout reward"),
        ("ode_success_rate", "ODE success rate"),
        ("mixed_group_fraction", "Mixed-group fraction"),
        ("zero_advantage_group_fraction", "Zero-advantage group fraction"),
        ("advantage_abs_mean", "Mean absolute advantage"),
        ("gradient_norm", "Policy gradient norm (beta=0)"),
    )

    for ax, (metric_name, title) in zip(axes.flat, panels, strict=True):
        ax.plot(
            collapse_history["iteration"],
            collapse_history[metric_name],
            label="Collapse: G=4",
        )
        ax.plot(
            recovery_history["iteration"],
            recovery_history[metric_name],
            label="Recovery: G=64",
        )
        ax.set_title(title)
        ax.set_xlabel("GRPO iteration")
        ax.grid(alpha=0.2)
        ax.legend()

    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def summarize_history(history: dict[str, list[float]]) -> dict[str, float]:
    """Return compact statistics used for console and experiment notes."""
    zero_gradient_rate = sum(
        gradient < 1e-12 for gradient in history["gradient_norm"]
    ) / len(history["gradient_norm"])
    return {
        "mean_reward": sum(history["rollout_reward"])
        / len(history["rollout_reward"]),
        "mean_mixed_group_fraction": sum(history["mixed_group_fraction"])
        / len(history["mixed_group_fraction"]),
        "mean_zero_advantage_group_fraction": sum(
            history["zero_advantage_group_fraction"]
        )
        / len(history["zero_advantage_group_fraction"]),
        "mean_advantage_abs": sum(history["advantage_abs_mean"])
        / len(history["advantage_abs_mean"]),
        "mean_gradient_norm": sum(history["gradient_norm"])
        / len(history["gradient_norm"]),
        "zero_gradient_iteration_rate": zero_gradient_rate,
        "initial_ode_success": history["ode_success_rate"][0],
        "final_ode_success": history["ode_success_rate"][-1],
        "final_sde_success": history["sde_success_rate"][-1],
    }


def main() -> None:
    print(f"Using device: {DEVICE}")
    reference_state = load_or_train_reference_state()
    reward_radius, calibration = calibrate_reward_radius(reference_state)
    calibrated_success = calibration[reward_radius]

    print("\nReward-radius calibration:")
    for radius, success_rate in calibration.items():
        print(f"  radius={radius:>5.3f} | success={success_rate:.4f}")
    print(
        f"Selected radius={reward_radius:.3f} "
        f"(success={calibrated_success:.4f})"
    )

    common = {
        "num_groups": 16,
        "denoising_steps": 20,
        "eta": 0.5,
        # Isolate policy-gradient collapse. Otherwise KL can create a total
        # gradient even when every reward advantage is zero.
        "beta": 0.0,
        "learning_rate": 1e-4,
        "ppo_epochs": 2,
        "training_iterations": 50,
        "evaluation_samples": 2048,
        "evaluation_interval": 5,
        "reward_radius": reward_radius,
    }
    collapse_config = GRPOConfig(group_size=4, **common)
    recovery_config = GRPOConfig(group_size=64, **common)

    print("\nExpected mixed-group probabilities at calibrated success rate:")
    for group_size in (collapse_config.group_size, recovery_config.group_size):
        probability = mixed_group_probability(calibrated_success, group_size)
        print(f"  G={group_size:>2d} | P(mixed)={probability:.4f}")

    print("\n=== Collapse condition: G=4 ===")
    collapse_history = run_condition(
        reference_state,
        collapse_config,
        seed=2026,
    )

    print("\n=== Recovery condition: G=64 ===")
    recovery_history = run_condition(
        reference_state,
        recovery_config,
        seed=2026,
    )

    figure_path = save_comparison_figure(
        collapse_history,
        recovery_history,
        FIGURE_PATH,
    )
    collapse_summary = summarize_history(collapse_history)
    recovery_summary = summarize_history(recovery_history)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "reward_radius": reward_radius,
            "calibration": calibration,
            "collapse_config": asdict(collapse_config),
            "recovery_config": asdict(recovery_config),
            "collapse_history": collapse_history,
            "recovery_history": recovery_history,
            "collapse_summary": collapse_summary,
            "recovery_summary": recovery_summary,
        },
        HISTORY_PATH,
    )

    print("\nCollapse summary:", collapse_summary)
    print("Recovery summary:", recovery_summary)
    print(f"Saved comparison figure to: {figure_path}")
    print(f"Saved histories to: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
