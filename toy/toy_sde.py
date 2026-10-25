"""Day 2: turn a learned CFM velocity field into an SDE sampler."""

import math

import torch
from torch import nn

if __package__:
    from toy.toy_flow import D, DEVICE, VNet, ode_sample, sample_data, train_cfm
    from toy.visualization import save_sample_comparison
else:
    # Also support direct imports made by: python toy/toy_grpo.py
    from toy_flow import D, DEVICE, VNet, ode_sample, sample_data, train_cfm
    from visualization import save_sample_comparison


def score_from_velocity(
    x: torch.Tensor,
    velocity: torch.Tensor,
    time: torch.Tensor,
) -> torch.Tensor:
    """Compute score = -(x + (1-t) * v) / t for the linear CFM path."""
    if x.shape != velocity.shape:
        raise ValueError("x and velocity must have the same shape")
    if time.ndim != 2 or time.shape != (x.shape[0], 1):
        raise ValueError(
            f"time must have shape {(x.shape[0], 1)}, got {tuple(time.shape)}"
        )
    if torch.any(time <= 0):
        raise ValueError("score_from_velocity requires time > 0")

    return -(x + (1 - time) * velocity) / time


def sde_step(
    model: nn.Module,
    x: torch.Tensor,
    time: float,
    eta: float,
    h: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Gaussian transition mean and scalar standard deviation."""
    if not 0 < time <= 1:
        raise ValueError("time must be in (0, 1]")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if h <= 0:
        raise ValueError("h must be positive")

    time_tensor = torch.full(
        (x.shape[0], 1),
        time,
        dtype=x.dtype,
        device=x.device,
    )

    velocity = model(x, time_tensor)
    score = score_from_velocity(x, velocity, time_tensor)

    # g_t = eta * sqrt(t)
    diffusion = eta * math.sqrt(time)

    # Reverse-time drift: mu_r = v - 1/2 * g_t^2 * score.
    reverse_drift = velocity - 0.5 * diffusion**2 * score

    # Time goes from t to t-h, hence the deterministic increment is -h * mu_r.
    mean = x - h * reverse_drift

    # Brownian increment sqrt(h) * epsilon gives std = g_t * sqrt(h).
    std = x.new_tensor(diffusion * math.sqrt(h))
    return mean, std


def sample_sde_transition(
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Sample x_next = mean + std * epsilon, epsilon ~ N(0, I)."""
    if torch.any(std < 0):
        raise ValueError("std must be non-negative")
    return mean + std * torch.randn_like(mean)


@torch.no_grad()
def sde_rollout(
    model: nn.Module,
    n: int = 512,
    steps: int = 100,
    eta: float = 0.5,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a full trajectory with shape (batch, steps + 1, D)."""
    if n <= 0:
        raise ValueError("n must be positive")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if eta < 0:
        raise ValueError("eta must be non-negative")

    model.eval()
    device = next(model.parameters()).device

    if initial_noise is None:
        x = torch.randn(n, D, device=device)
    else:
        if initial_noise.shape != (n, D):
            raise ValueError(
                f"initial_noise must have shape {(n, D)}, "
                f"got {tuple(initial_noise.shape)}"
            )
        x = initial_noise.detach().clone().to(device)

    h = 1.0 / steps
    trajectory = [x]

    for step in range(steps):
        time = 1 - step * h
        mean, std = sde_step(
            model=model,
            x=x,
            time=time,
            eta=eta,
            h=h,
        )
        x = sample_sde_transition(mean, std)
        trajectory.append(x)

    return torch.stack(trajectory, dim=1)


def gaussian_log_prob(
    x_next: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Log N(x_next; mean, std^2 I), summed over state dimensions."""
    if x_next.shape != mean.shape:
        raise ValueError("x_next and mean must have the same shape")
    if torch.any(std <= 0):
        raise ValueError("std must be positive when evaluating log-probability")

    variance = std.square()
    log_prob_per_dimension = -0.5 * (
        (x_next - mean).square() / variance
        + torch.log(2 * math.pi * variance)
    )
    return log_prob_per_dimension.sum(dim=-1)


def trajectory_log_probs(
    model: nn.Module,
    trajectory: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    """Evaluate every transition; return shape (batch, steps)."""
    if trajectory.ndim != 3 or trajectory.shape[-1] != D:
        raise ValueError(
            f"trajectory must have shape (batch, steps + 1, {D}), "
            f"got {tuple(trajectory.shape)}"
        )
    if trajectory.shape[1] < 2:
        raise ValueError("trajectory must contain at least two states")
    if eta <= 0:
        raise ValueError("eta must be positive when evaluating log-probability")

    # The sampled states are fixed data here. Gradients should only flow through
    # the newly evaluated transition means, and therefore into model parameters.
    fixed_trajectory = trajectory.detach()
    steps = fixed_trajectory.shape[1] - 1
    h = 1.0 / steps
    log_probs = []

    for step in range(steps):
        time = 1 - step * h
        x_current = fixed_trajectory[:, step]
        x_next = fixed_trajectory[:, step + 1]
        mean, std = sde_step(
            model=model,
            x=x_current,
            time=time,
            eta=eta,
            h=h,
        )
        log_probs.append(
            gaussian_log_prob(
                x_next=x_next,
                mean=mean,
                std=std,
            )
        )

    return torch.stack(log_probs, dim=1)


def main() -> None:
    torch.manual_seed(0)
    print(f"Using device: {DEVICE}")

    model = VNet().to(DEVICE)
    losses = train_cfm(model, steps=5000, batch_size=512, lr=1e-3)

    sample_count = 5000
    integration_steps = 100
    eta = 0.5

    initial_noise = torch.randn(sample_count, D, device=DEVICE)
    target_samples = sample_data(sample_count)
    ode_samples = ode_sample(
        model,
        n=sample_count,
        steps=integration_steps,
        initial_noise=initial_noise,
    )
    sde_trajectory = sde_rollout(
        model,
        n=sample_count,
        steps=integration_steps,
        eta=eta,
        initial_noise=initial_noise,
    )

    output_path = save_sample_comparison(
        panels={
            "Target": target_samples,
            f"ODE N={integration_steps}": ode_samples,
            f"SDE N={integration_steps}, eta={eta}": sde_trajectory[:, -1],
        },
        losses=losses,
        output_path="results/day2_sde_endpoints.png",
    )

    example_log_probs = trajectory_log_probs(
        model=model,
        trajectory=sde_trajectory[:64],
        eta=eta,
    )
    print(f"Trajectory shape: {tuple(sde_trajectory.shape)}")
    print(f"Transition log-prob shape: {tuple(example_log_probs.shape)}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
