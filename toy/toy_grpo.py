"""Day 3: optimize the toy SDE policy with GRPO/PPO clipping."""

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
    from toy.toy_sde import sde_rollout, trajectory_log_probs
    from toy.visualization import save_sample_comparison
else:
    # Also support: python toy/toy_grpo.py
    from toy_flow import D, DEVICE, VNet, ode_sample, train_cfm
    from toy_sde import sde_rollout, trajectory_log_probs
    from visualization import save_sample_comparison


# ============================================================
# 1. Configuration
# ============================================================


@dataclass
class GRPOConfig:
    """Hyperparameters for the toy GRPO experiment."""

    # Number of independent GRPO groups.
    num_groups: int = 32

    # Number of trajectories sampled for the same external condition/task.
    group_size: int = 8

    # Number of SDE denoising transitions in one trajectory.
    denoising_steps: int = 20

    # Diffusion schedule:
    #
    #     g(t) = eta * sqrt(t)
    #
    # Therefore one Euler-Maruyama transition has:
    #
    #     std_k = g(t_k) * sqrt(h)
    #           = eta * sqrt(t_k * h)
    eta: float = 0.5

    # PPO clipping parameter:
    #
    #     clip(rho, 1-epsilon, 1+epsilon)
    clip_epsilon: float = 0.2

    # Coefficient of KL-to-reference regularization.
    beta: float = 0.01

    # Learning rate for GRPO.
    learning_rate: float = 1e-4

    # Number of gradient updates performed on one rollout batch.
    ppo_epochs: int = 4

    # Number of rollout/update iterations.
    training_iterations: int = 200

    # Gradient clipping.
    max_grad_norm: float = 1.0

    # By default, use:
    #
    #     A_i = r_i - group_mean
    #
    # without dividing by group standard deviation.
    use_advantage_std: bool = False

    # Reward definition:
    #
    #     r = 1[||x_final - target_center|| < reward_radius]
    target_center_x: float = 4.0
    target_center_y: float = 0.0
    reward_radius: float = 1.0

    # Number of samples used for periodic evaluation.
    evaluation_samples: int = 4096

    # Evaluate every N GRPO iterations.
    evaluation_interval: int = 10


# ============================================================
# 2. Basic utility functions
# ============================================================


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Put a model in evaluation mode and disable parameter gradients.

    We use this for:

    1. old_model:
       the frozen behavior policy that generated the current rollout;

    2. reference_model:
       the frozen original SFT/CFM policy used for KL regularization.
    """
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def assert_all_finite(name: str, tensor: torch.Tensor) -> None:
    """Fail immediately if a tensor contains NaN or Inf."""
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")


def parameter_gradient_norm(model: nn.Module) -> float:
    """Compute the total L2 norm of all available parameter gradients."""
    squared_norm = 0.0

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        squared_norm += parameter.grad.detach().square().sum().item()

    return math.sqrt(squared_norm)


# ============================================================
# 3. Terminal reward
# ============================================================


def terminal_reward(
    endpoints: torch.Tensor,
    target_center: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """
    Binary terminal reward.

    Formula:

        r_i =
        1, if ||x_final_i - c||_2 < radius
        0, otherwise

    Shape:
        endpoints: (batch, D)
        reward:    (batch,)
    """
    if endpoints.ndim != 2 or endpoints.shape[-1] != D:
        raise ValueError(
            f"endpoints must have shape (batch, {D}), "
            f"got {tuple(endpoints.shape)}"
        )

    if target_center.shape != (D,):
        raise ValueError(
            f"target_center must have shape ({D},), "
            f"got {tuple(target_center.shape)}"
        )

    if radius <= 0:
        raise ValueError("radius must be positive")

    distances = torch.linalg.vector_norm(
        endpoints - target_center,
        dim=-1,
    )

    return (distances < radius).to(dtype=endpoints.dtype)


# ============================================================
# 4. Group-relative advantages
# ============================================================


def compute_group_advantages(
    rewards: torch.Tensor,
    num_groups: int,
    group_size: int,
    use_std: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute GRPO group-relative advantages.

    For group g:

        mean_g = (1/G) * sum_j r_(g,j)

        A_(g,i) = r_(g,i) - mean_g

    By default we do NOT divide by the group standard deviation.

    If use_std=True:

        A_(g,i) =
            [r_(g,i) - mean_g] / [std_g + eps]

    Output shape:
        (batch,)

    Important:
        The returned advantages are detached because rewards and advantages
        are treated as fixed rollout data.
    """
    expected_batch_size = num_groups * group_size

    if rewards.shape != (expected_batch_size,):
        raise ValueError(
            f"rewards must have shape {(expected_batch_size,)}, "
            f"got {tuple(rewards.shape)}"
        )

    grouped_rewards = rewards.reshape(num_groups, group_size)

    group_means = grouped_rewards.mean(
        dim=1,
        keepdim=True,
    )

    advantages = grouped_rewards - group_means

    if use_std:
        # unbiased=False is appropriate for normalization within the sampled
        # finite group.
        group_stds = grouped_rewards.std(
            dim=1,
            keepdim=True,
            unbiased=False,
        )

        advantages = advantages / (group_stds + eps)

    return advantages.reshape(-1).detach()


def compute_group_diagnostics(
    rewards: torch.Tensor,
    num_groups: int,
    group_size: int,
) -> dict[str, torch.Tensor]:
    """Measure which binary-reward groups contain a learning signal."""
    expected_batch_size = num_groups * group_size
    if rewards.shape != (expected_batch_size,):
        raise ValueError(
            f"rewards must have shape {(expected_batch_size,)}, "
            f"got {tuple(rewards.shape)}"
        )

    grouped_rewards = rewards.reshape(num_groups, group_size)
    success_counts = grouped_rewards.sum(dim=1)
    all_failure = success_counts == 0
    all_success = success_counts == group_size
    mixed = ~(all_failure | all_success)

    return {
        "mixed_group_fraction": mixed.float().mean().detach(),
        "all_failure_group_fraction": all_failure.float().mean().detach(),
        "all_success_group_fraction": all_success.float().mean().detach(),
        "zero_advantage_group_fraction": (
            all_failure | all_success
        ).float().mean().detach(),
    }


# ============================================================
# 5. Sample one GRPO rollout batch
# ============================================================


@torch.no_grad()
def collect_rollout_batch(
    old_model: nn.Module,
    reference_model: nn.Module,
    config: GRPOConfig,
    target_center: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Generate one complete GRPO rollout batch.

    A group represents the same external condition. Each trajectory keeps
    independent policy randomness: independent initial Gaussian noise and
    independent Brownian increments. In the unconditional toy there is no
    explicit condition tensor, so grouping only determines the reward baseline.

    Returned tensors:

        trajectory:
            (batch, denoising_steps + 1, D)

        old_log_probs:
            (batch, denoising_steps)

        reference_log_probs:
            (batch, denoising_steps)

        rewards:
            (batch,)

        advantages:
            (batch,)
    """
    batch_size = config.num_groups * config.group_size

    # --------------------------------------------------------
    # One independent initial flow-noise state per trajectory.
    #
    # Shape:
    #     (num_groups * group_size, D)
    #
    # In a conditional VLA, trajectories in one group share the same robot
    # observation/task condition, not the same sampled policy noise.
    # --------------------------------------------------------
    device = next(old_model.parameters()).device
    initial_noise = torch.randn(
        batch_size,
        D,
        device=device,
    )

    # --------------------------------------------------------
    # Sample trajectories using the frozen old policy.
    #
    # Reverse Euler-Maruyama transition:
    #
    #     x_(k+1)
    #       ~ N(
    #           x_k - h * mu_old(x_k, t_k),
    #           g(t_k)^2 * h * I
    #         )
    #
    # Shape:
    #     (batch, denoising_steps + 1, D)
    # --------------------------------------------------------
    trajectory = sde_rollout(
        model=old_model,
        n=batch_size,
        steps=config.denoising_steps,
        eta=config.eta,
        initial_noise=initial_noise,
    )

    # --------------------------------------------------------
    # Re-evaluate the sampled fixed trajectory under old_model.
    #
    # Each entry is:
    #
    #     log pi_old(x_(k+1) | x_k)
    #
    # Shape:
    #     (batch, denoising_steps)
    # --------------------------------------------------------
    old_log_probs = trajectory_log_probs(
        model=old_model,
        trajectory=trajectory,
        eta=config.eta,
    )

    # --------------------------------------------------------
    # Evaluate the same fixed trajectory under the frozen
    # original SFT/CFM reference policy.
    #
    # Shape:
    #     (batch, denoising_steps)
    # --------------------------------------------------------
    reference_log_probs = trajectory_log_probs(
        model=reference_model,
        trajectory=trajectory,
        eta=config.eta,
    )

    endpoints = trajectory[:, -1]

    rewards = terminal_reward(
        endpoints=endpoints,
        target_center=target_center,
        radius=config.reward_radius,
    )

    advantages = compute_group_advantages(
        rewards=rewards,
        num_groups=config.num_groups,
        group_size=config.group_size,
        use_std=config.use_advantage_std,
    )
    group_diagnostics = compute_group_diagnostics(
        rewards=rewards,
        num_groups=config.num_groups,
        group_size=config.group_size,
    )

    return {
        "trajectory": trajectory.detach(),
        "old_log_probs": old_log_probs.detach(),
        "reference_log_probs": reference_log_probs.detach(),
        "rewards": rewards.detach(),
        "advantages": advantages.detach(),
        **group_diagnostics,
    }


# ============================================================
# 6. PPO-clipped GRPO loss
# ============================================================


def k3_kl_penalty(
    new_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Return pointwise k3 for KL(current || reference).

    With r = pi_ref / pi_theta and u = log(r):

        k3 = (r - 1) - log(r) = exp(u) - u - 1.
    """
    if new_log_probs.shape != reference_log_probs.shape:
        raise ValueError(
            "new_log_probs and reference_log_probs must have the same shape"
        )

    log_reference_over_current = reference_log_probs - new_log_probs
    return (
        torch.exp(log_reference_over_current)
        - log_reference_over_current
        - 1.0
    )


def compute_grpo_loss(
    model: nn.Module,
    rollout_batch: dict[str, torch.Tensor],
    config: GRPOConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Compute the GRPO/PPO loss on one fixed rollout batch.

    The sampled trajectory, old log-probs, reference log-probs and
    advantages are all fixed data.

    The only differentiable quantity is:

        new_log_probs = log pi_theta(x_(k+1) | x_k)

    Gradients therefore flow through:

        new_log_prob
            -> Gaussian transition mean
            -> reverse SDE drift
            -> velocity network
            -> theta
    """
    trajectory = rollout_batch["trajectory"]
    old_log_probs = rollout_batch["old_log_probs"]
    reference_log_probs = rollout_batch["reference_log_probs"]
    advantages = rollout_batch["advantages"]

    # --------------------------------------------------------
    # Recompute transition log-probs under the CURRENT policy.
    #
    # Shape:
    #     (batch, denoising_steps)
    # --------------------------------------------------------
    new_log_probs = trajectory_log_probs(
        model=model,
        trajectory=trajectory,
        eta=config.eta,
    )

    assert_all_finite("new_log_probs", new_log_probs)

    # --------------------------------------------------------
    # Broadcast one episode advantage to every denoising step.
    #
    # advantages:
    #     (batch,)
    #
    # step_advantages:
    #     (batch, 1)
    #
    # It broadcasts against:
    #     (batch, denoising_steps)
    #
    # Therefore all denoising transitions of one trajectory receive
    # the same terminal-reward advantage.
    # --------------------------------------------------------
    step_advantages = advantages[:, None]

    # --------------------------------------------------------
    # PPO importance ratio:
    #
    #     rho_(i,k)(theta)
    #       = pi_theta(x_(k+1) | x_k)
    #         --------------------------------
    #         pi_old(x_(k+1) | x_k)
    #
    # In log space:
    #
    #     rho = exp(log pi_theta - log pi_old)
    # --------------------------------------------------------
    log_ratio_old = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio_old)

    assert_all_finite("ratio", ratio)

    # --------------------------------------------------------
    # Unclipped PPO objective:
    #
    #     L_unclipped = rho * A
    # --------------------------------------------------------
    unclipped_objective = ratio * step_advantages

    # --------------------------------------------------------
    # Clipped PPO objective:
    #
    #     L_clipped =
    #       clip(rho, 1-epsilon, 1+epsilon) * A
    # --------------------------------------------------------
    clipped_ratio = torch.clamp(
        ratio,
        min=1.0 - config.clip_epsilon,
        max=1.0 + config.clip_epsilon,
    )

    clipped_objective = clipped_ratio * step_advantages

    # --------------------------------------------------------
    # PPO maximizes:
    #
    #     E[min(rho A, clip(rho) A)]
    #
    # PyTorch optimizers minimize loss, therefore:
    #
    #     policy_loss =
    #       -E[min(rho A, clip(rho) A)]
    # --------------------------------------------------------
    policy_loss = -torch.minimum(
        unclipped_objective,
        clipped_objective,
    ).mean()

    # --------------------------------------------------------
    # k3 KL-to-reference penalty.
    #
    # Project convention:
    #
    #     r = pi_ref / pi_theta
    #     u = log(r) = log pi_ref - log pi_theta
    #     KL_k3 = exp(u) - u - 1
    #
    # It satisfies:
    #
    #     KL_k3 >= 0
    #
    # and equals zero when:
    #
    #     pi_theta = pi_ref
    # --------------------------------------------------------
    kl_per_transition = k3_kl_penalty(
        new_log_probs=new_log_probs,
        reference_log_probs=reference_log_probs,
    )

    kl_loss = kl_per_transition.mean()

    # --------------------------------------------------------
    # Total objective:
    #
    #     L(theta)
    #       = L_PPO(theta)
    #         + beta * KL_k3(theta || reference)
    # --------------------------------------------------------
    total_loss = policy_loss + config.beta * kl_loss

    # --------------------------------------------------------
    # Monitoring metrics.
    # --------------------------------------------------------
    clip_fraction = (
        (ratio - 1.0).abs() > config.clip_epsilon
    ).float().mean()

    approximate_old_kl = (
        old_log_probs - new_log_probs
    ).mean()

    metrics = {
        "total_loss": total_loss.detach(),
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl_loss.detach(),
        "mean_ratio": ratio.mean().detach(),
        "clip_fraction": clip_fraction.detach(),
        "approximate_old_kl": approximate_old_kl.detach(),
        "mean_new_log_prob": new_log_probs.mean().detach(),
    }

    return total_loss, metrics


# ============================================================
# 7. One GRPO update on one rollout batch
# ============================================================


def update_policy(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout_batch: dict[str, torch.Tensor],
    config: GRPOConfig,
) -> dict[str, float]:
    """
    Perform multiple PPO epochs over the same fixed rollout batch.

    old_log_probs remain frozen across these epochs.

    This is essential: PPO compares the changing current policy against
    the single behavior policy that generated the rollout.
    """
    model.train()

    final_metrics: dict[str, float] = {}

    for _ in range(config.ppo_epochs):
        optimizer.zero_grad(set_to_none=True)

        loss, tensor_metrics = compute_grpo_loss(
            model=model,
            rollout_batch=rollout_batch,
            config=config,
        )

        assert_all_finite("GRPO loss", loss)

        loss.backward()

        gradient_norm_before_clipping = parameter_gradient_norm(model)

        if not math.isfinite(gradient_norm_before_clipping):
            raise FloatingPointError(
                "Gradient norm is NaN or Inf"
            )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.max_grad_norm,
        )

        optimizer.step()

        final_metrics = {
            name: value.item()
            for name, value in tensor_metrics.items()
        }

        final_metrics["gradient_norm"] = (
            gradient_norm_before_clipping
        )

    return final_metrics


# ============================================================
# 8. Evaluation
# ============================================================


@torch.no_grad()
def evaluate_policy(
    model: nn.Module,
    config: GRPOConfig,
    target_center: torch.Tensor,
) -> dict[str, float]:
    """
    Evaluate both:

    1. deterministic ODE sampling;
    2. stochastic SDE sampling.

    Final deployment is expected to use ODE sampling, but SDE evaluation
    is useful for observing the train-SDE/eval-ODE gap.
    """
    model.eval()

    initial_noise = torch.randn(
        config.evaluation_samples,
        D,
        device=DEVICE,
    )

    ode_endpoints = ode_sample(
        model=model,
        n=config.evaluation_samples,
        steps=config.denoising_steps,
        initial_noise=initial_noise,
    )

    ode_rewards = terminal_reward(
        endpoints=ode_endpoints,
        target_center=target_center,
        radius=config.reward_radius,
    )

    sde_trajectory = sde_rollout(
        model=model,
        n=config.evaluation_samples,
        steps=config.denoising_steps,
        eta=config.eta,
        initial_noise=initial_noise,
    )

    sde_rewards = terminal_reward(
        endpoints=sde_trajectory[:, -1],
        target_center=target_center,
        radius=config.reward_radius,
    )

    return {
        "ode_success_rate": ode_rewards.mean().item(),
        "sde_success_rate": sde_rewards.mean().item(),
    }


# ============================================================
# 9. Save training curves
# ============================================================


def save_training_curves(
    history: dict[str, list[float]],
    output_path: str | Path,
) -> Path:
    """Save Day 3 training diagnostics."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    updates = history["iteration"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11, 8),
    )

    axes[0, 0].plot(
        updates,
        history["rollout_reward"],
        label="Rollout reward",
    )
    axes[0, 0].plot(
        updates,
        history["ode_success_rate"],
        label="ODE success",
    )
    axes[0, 0].plot(
        updates,
        history["sde_success_rate"],
        label="SDE success",
    )
    axes[0, 0].set_title("Reward and success rate")
    axes[0, 0].set_xlabel("GRPO iteration")
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].plot(
        updates,
        history["total_loss"],
        label="Total loss",
    )
    axes[0, 1].plot(
        updates,
        history["policy_loss"],
        label="Policy loss",
    )
    axes[0, 1].set_title("Training loss")
    axes[0, 1].set_xlabel("GRPO iteration")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    axes[1, 0].plot(
        updates,
        history["kl_loss"],
        label="Reference KL",
    )
    axes[1, 0].plot(
        updates,
        history["approximate_old_kl"],
        label="Old-policy KL estimate",
    )
    axes[1, 0].set_title("KL diagnostics")
    axes[1, 0].set_xlabel("GRPO iteration")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.2)

    axes[1, 1].plot(
        updates,
        history["mean_ratio"],
        label="Mean ratio",
    )
    axes[1, 1].plot(
        updates,
        history["clip_fraction"],
        label="Clip fraction",
    )
    axes[1, 1].plot(
        updates,
        history["gradient_norm"],
        label="Gradient norm",
    )
    axes[1, 1].set_title("PPO optimization diagnostics")
    axes[1, 1].set_xlabel("GRPO iteration")
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        destination,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    return destination


# ============================================================
# 10. Complete training loop
# ============================================================


def train_grpo(
    model: nn.Module,
    reference_model: nn.Module,
    config: GRPOConfig,
) -> dict[str, list[float]]:
    """
    Full GRPO loop.

    At every iteration:

    1. copy current model -> old_model;
    2. use old_model to collect trajectories;
    3. compute terminal rewards;
    4. compute group-relative advantages;
    5. hold rollout data fixed;
    6. optimize current model with PPO clipping and reference KL.
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    target_center = torch.tensor(
        [
            config.target_center_x,
            config.target_center_y,
        ],
        device=DEVICE,
        dtype=next(model.parameters()).dtype,
    )

    history: dict[str, list[float]] = {
        "iteration": [],
        "rollout_reward": [],
        "advantage_abs_mean": [],
        "total_loss": [],
        "policy_loss": [],
        "kl_loss": [],
        "mean_ratio": [],
        "clip_fraction": [],
        "approximate_old_kl": [],
        "gradient_norm": [],
        "mixed_group_fraction": [],
        "all_failure_group_fraction": [],
        "all_success_group_fraction": [],
        "zero_advantage_group_fraction": [],
        "ode_success_rate": [],
        "sde_success_rate": [],
    }

    initial_evaluation = evaluate_policy(
        model=model,
        config=config,
        target_center=target_center,
    )

    print(
        "\nInitial policy:"
        f" ODE success={initial_evaluation['ode_success_rate']:.4f},"
        f" SDE success={initial_evaluation['sde_success_rate']:.4f}"
    )

    latest_evaluation = initial_evaluation

    for iteration in range(1, config.training_iterations + 1):
        # ----------------------------------------------------
        # PPO behavior policy:
        #
        # theta_old <- theta
        #
        # This snapshot remains fixed while the current policy is
        # updated on this rollout batch.
        # ----------------------------------------------------
        old_model = freeze_model(
            copy.deepcopy(model)
        )

        rollout_batch = collect_rollout_batch(
            old_model=old_model,
            reference_model=reference_model,
            config=config,
            target_center=target_center,
        )

        rewards = rollout_batch["rewards"]
        advantages = rollout_batch["advantages"]

        rollout_reward = rewards.mean().item()
        advantage_abs_mean = advantages.abs().mean().item()
        mixed_group_fraction = rollout_batch[
            "mixed_group_fraction"
        ].item()

        # Binary group rewards may occasionally all be identical within
        # every group, which gives all-zero GRPO advantages.
        #
        # The update is mathematically valid but contains no reward signal.
        metrics = update_policy(
            model=model,
            optimizer=optimizer,
            rollout_batch=rollout_batch,
            config=config,
        )

        if (
            iteration == 1
            or iteration % config.evaluation_interval == 0
            or iteration == config.training_iterations
        ):
            latest_evaluation = evaluate_policy(
                model=model,
                config=config,
                target_center=target_center,
            )

        history["iteration"].append(float(iteration))
        history["rollout_reward"].append(rollout_reward)
        history["advantage_abs_mean"].append(
            advantage_abs_mean
        )
        history["total_loss"].append(
            metrics["total_loss"]
        )
        history["policy_loss"].append(
            metrics["policy_loss"]
        )
        history["kl_loss"].append(
            metrics["kl_loss"]
        )
        history["mean_ratio"].append(
            metrics["mean_ratio"]
        )
        history["clip_fraction"].append(
            metrics["clip_fraction"]
        )
        history["approximate_old_kl"].append(
            metrics["approximate_old_kl"]
        )
        history["gradient_norm"].append(
            metrics["gradient_norm"]
        )
        for diagnostic_name in (
            "mixed_group_fraction",
            "all_failure_group_fraction",
            "all_success_group_fraction",
            "zero_advantage_group_fraction",
        ):
            history[diagnostic_name].append(
                rollout_batch[diagnostic_name].item()
            )
        history["ode_success_rate"].append(
            latest_evaluation["ode_success_rate"]
        )
        history["sde_success_rate"].append(
            latest_evaluation["sde_success_rate"]
        )

        if (
            iteration == 1
            or iteration % config.evaluation_interval == 0
        ):
            print(
                f"Iteration {iteration:4d} | "
                f"reward={rollout_reward:.4f} | "
                f"|A|={advantage_abs_mean:.4f} | "
                f"mixed={mixed_group_fraction:.4f} | "
                f"loss={metrics['total_loss']:.6f} | "
                f"policy={metrics['policy_loss']:.6f} | "
                f"KL={metrics['kl_loss']:.6f} | "
                f"ratio={metrics['mean_ratio']:.4f} | "
                f"clip={metrics['clip_fraction']:.4f} | "
                f"grad={metrics['gradient_norm']:.4f} | "
                f"ODE success="
                f"{latest_evaluation['ode_success_rate']:.4f} | "
                f"SDE success="
                f"{latest_evaluation['sde_success_rate']:.4f}"
            )

    return history


# ============================================================
# 11. Main experiment
# ============================================================


def main() -> None:
    torch.manual_seed(0)

    print(f"Using device: {DEVICE}")

    config = GRPOConfig()

    # --------------------------------------------------------
    # Step A: SFT / CFM pretraining
    #
    # Train:
    #
    #     v_theta(x_t, t) ≈ z - a
    #
    # where:
    #
    #     x_t = (1-t)a + tz
    #
    # This produces the initial Flow Matching policy.
    # --------------------------------------------------------
    model = VNet().to(DEVICE)

    print("\nTraining the initial CFM policy...")

    cfm_losses = train_cfm(
        model=model,
        steps=5000,
        batch_size=512,
        lr=1e-3,
    )

    # --------------------------------------------------------
    # Step B: Freeze the original SFT/CFM reference model.
    #
    # This model never changes during GRPO.
    # --------------------------------------------------------
    reference_model = freeze_model(
        copy.deepcopy(model)
    )

    # Save initial endpoints before RL.
    comparison_noise = torch.randn(
        5000,
        D,
        device=DEVICE,
    )

    initial_ode_endpoints = ode_sample(
        model=model,
        n=comparison_noise.shape[0],
        steps=config.denoising_steps,
        initial_noise=comparison_noise,
    )

    # --------------------------------------------------------
    # Step C: GRPO training.
    # --------------------------------------------------------
    history = train_grpo(
        model=model,
        reference_model=reference_model,
        config=config,
    )

    # --------------------------------------------------------
    # Step D: Compare pre-GRPO and post-GRPO ODE endpoints.
    # Use the same initial noise for a controlled comparison.
    # --------------------------------------------------------
    final_ode_endpoints = ode_sample(
        model=model,
        n=comparison_noise.shape[0],
        steps=config.denoising_steps,
        initial_noise=comparison_noise,
    )

    target_center = torch.tensor(
        [
            config.target_center_x,
            config.target_center_y,
        ],
        device=DEVICE,
    )

    reward_circle_angles = torch.linspace(
        0.0,
        2.0 * math.pi,
        256,
        device=DEVICE,
    )
    target_marker_samples = (
        target_center[None]
        + config.reward_radius
        * torch.stack(
            [
                torch.cos(reward_circle_angles),
                torch.sin(reward_circle_angles),
            ],
            dim=1,
        )
    )

    comparison_path = save_sample_comparison(
        panels={
            "Before GRPO: ODE": initial_ode_endpoints,
            "After GRPO: ODE": final_ode_endpoints,
            "Reward target": target_marker_samples,
        },
        losses=cfm_losses,
        output_path="results/day3_grpo_endpoints.png",
    )

    curve_path = save_training_curves(
        history=history,
        output_path="results/day3_grpo_training.png",
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "reference_state_dict": reference_model.state_dict(),
            # Save plain data rather than an instance that may be pickled as
            # __main__.GRPOConfig when this file is run as a script.
            "config": asdict(config),
            "history": history,
        },
        "results/day3_toy_grpo_checkpoint.pt",
    )

    print(f"\nSaved endpoint comparison to: {comparison_path}")
    print(f"Saved training curves to: {curve_path}")
    print(
        "Saved checkpoint to: "
        "results/day3_toy_grpo_checkpoint.pt"
    )


if __name__ == "__main__":
    main()
