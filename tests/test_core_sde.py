"""Model-independent tests for the production SDE kernel."""

import math

import pytest
import torch

from flowvla_rl.sde import gaussian_log_prob, marginal_preserving_transition


def test_eta_zero_is_smolvla_euler_step() -> None:
    torch.manual_seed(0)
    x_t = torch.randn(3, 5, 7)
    velocity = torch.randn_like(x_t)
    transition = marginal_preserving_transition(x_t, velocity, 0.7, 0.1, eta=0.0)
    torch.testing.assert_close(transition.mean, x_t - 0.1 * velocity)
    assert torch.count_nonzero(transition.std) == 0


def test_time_one_score_and_transition_formula() -> None:
    x_t = torch.randn(2, 4, 3)
    velocity = torch.randn_like(x_t)
    transition = marginal_preserving_transition(x_t, velocity, 1.0, 0.1, eta=0.5)
    torch.testing.assert_close(transition.score, -x_t)
    expected_drift = velocity + 0.5 * 0.5**2 * x_t
    torch.testing.assert_close(transition.reverse_drift, expected_drift)
    torch.testing.assert_close(
        transition.std,
        torch.full((2, 1, 1), 0.5 * math.sqrt(0.1)),
    )


def test_masked_log_prob_excludes_padded_dimensions_and_positions() -> None:
    sample = torch.tensor([[[1.0, 2.0, 99.0], [3.0, 99.0, 99.0]]])
    mean = torch.zeros_like(sample, requires_grad=True)
    std = torch.ones(1, 1, 1)
    mask = torch.tensor([[[True, True, False], [True, False, False]]])
    actual = gaussian_log_prob(sample, mean, std, mask=mask)
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(
        torch.tensor([1.0, 2.0, 3.0])
    ).sum()
    torch.testing.assert_close(actual, expected[None])
    actual.sum().backward()
    assert mean.grad is not None
    assert torch.count_nonzero(mean.grad[~mask]) == 0


def test_log_prob_matches_torch_distribution() -> None:
    torch.manual_seed(2)
    sample = torch.randn(4, 5, 7)
    mean = torch.randn_like(sample, requires_grad=True)
    std = torch.rand(4, 1, 1) + 0.1
    actual = gaussian_log_prob(sample, mean, std)
    expected = torch.distributions.Normal(mean, std).log_prob(sample).sum((1, 2))
    torch.testing.assert_close(actual, expected)


def test_invalid_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="time"):
        marginal_preserving_transition(torch.zeros(2, 3), torch.zeros(2, 3), 0.0, 0.1, 0.5)

