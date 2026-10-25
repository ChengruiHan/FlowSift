"""Formula-level tests for the Day 2 toy SDE implementation."""

import math

import torch

from toy.toy_flow import D, DEVICE, VNet
from toy.toy_sde import (
    gaussian_log_prob,
    score_from_velocity,
    sde_rollout,
    sde_step,
    trajectory_log_probs,
)


def make_model() -> VNet:
    torch.manual_seed(0)
    return VNet().to(DEVICE)


def test_score_at_time_one_is_standard_normal_score() -> None:
    model = make_model()
    x = torch.randn(32, D, device=DEVICE)
    time = torch.ones(32, 1, device=DEVICE)

    with torch.no_grad():
        velocity = model(x, time)
        score = score_from_velocity(x, velocity, time)

    assert torch.allclose(score, -x, atol=1e-6)


def test_eta_zero_reduces_transition_mean_to_ode_euler() -> None:
    model = make_model()
    x = torch.randn(32, D, device=DEVICE)
    time = 0.7
    h = 0.1
    time_tensor = torch.full((x.shape[0], 1), time, device=DEVICE)

    with torch.no_grad():
        velocity = model(x, time_tensor)
        expected_mean = x - h * velocity
        mean, std = sde_step(model, x, time=time, eta=0.0, h=h)

    assert torch.allclose(mean, expected_mean, atol=1e-6)
    assert std.item() == 0.0


def test_transition_standard_deviation_formula() -> None:
    model = make_model()
    x = torch.randn(16, D, device=DEVICE)
    time, eta, h = 0.8, 0.5, 0.1

    with torch.no_grad():
        _, std = sde_step(model, x, time=time, eta=eta, h=h)

    assert math.isclose(
        std.item(),
        eta * math.sqrt(time * h),
        rel_tol=1e-6,
    )


def test_reverse_drift_uses_minus_score_correction() -> None:
    """Regression test for the reverse-SDE drift sign."""
    model = make_model()
    x = torch.randn(32, D, device=DEVICE)
    time, eta, h = 0.6, 0.8, 0.05
    time_tensor = torch.full((x.shape[0], 1), time, device=DEVICE)

    with torch.no_grad():
        velocity = model(x, time_tensor)
        score = score_from_velocity(x, velocity, time_tensor)
        diffusion_squared = eta**2 * time
        expected_reverse_drift = velocity - 0.5 * diffusion_squared * score
        expected_mean = x - h * expected_reverse_drift

        wrong_reverse_drift = velocity + 0.5 * diffusion_squared * score
        wrong_mean = x - h * wrong_reverse_drift
        actual_mean, _ = sde_step(model, x, time=time, eta=eta, h=h)

    torch.testing.assert_close(actual_mean, expected_mean)
    assert not torch.allclose(actual_mean, wrong_mean)


def test_rollout_shape_and_values() -> None:
    trajectory = sde_rollout(make_model(), n=17, steps=6, eta=0.5)

    assert trajectory.shape == (17, 7, D)
    assert torch.isfinite(trajectory).all()


def test_manual_gaussian_log_prob_matches_torch() -> None:
    model = make_model()
    x = torch.randn(16, D, device=DEVICE)

    with torch.no_grad():
        mean, std = sde_step(model, x, time=0.8, eta=0.5, h=0.1)
        x_next = mean + std * torch.randn_like(x)
        actual = gaussian_log_prob(x_next, mean, std)
        expected = torch.distributions.Normal(mean, std).log_prob(x_next).sum(-1)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_trajectory_log_probs_have_gradient_to_model() -> None:
    model = make_model()
    trajectory = sde_rollout(model, n=32, steps=6, eta=0.5)
    log_probs = trajectory_log_probs(model, trajectory, eta=0.5)

    assert log_probs.shape == (32, 6)
    assert torch.isfinite(log_probs).all()
    assert log_probs.requires_grad

    (-log_probs.mean()).backward()
    gradient = model.net[-1].bias.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_trajectory_log_prob_gradient_matches_finite_difference() -> None:
    model = make_model()
    trajectory = sde_rollout(model, n=32, steps=6, eta=0.5)

    model.zero_grad(set_to_none=True)
    objective = trajectory_log_probs(model, trajectory, eta=0.5).mean()
    objective.backward()

    parameter = model.net[-1].bias
    flat_parameter = parameter.flatten()
    flat_gradient = parameter.grad.flatten()
    parameter_index = flat_gradient.abs().argmax().item()
    auto_gradient = flat_gradient[parameter_index].item()

    epsilon = 1e-3
    original_value = flat_parameter[parameter_index].item()
    try:
        with torch.no_grad():
            flat_parameter[parameter_index] = original_value + epsilon
            loss_plus = trajectory_log_probs(
                model, trajectory, eta=0.5
            ).mean().item()

            flat_parameter[parameter_index] = original_value - epsilon
            loss_minus = trajectory_log_probs(
                model, trajectory, eta=0.5
            ).mean().item()
    finally:
        with torch.no_grad():
            flat_parameter[parameter_index] = original_value

    numeric_gradient = (loss_plus - loss_minus) / (2 * epsilon)
    relative_error = abs(auto_gradient - numeric_gradient) / max(
        1e-8,
        abs(auto_gradient) + abs(numeric_gradient),
    )
    assert relative_error < 1e-2
