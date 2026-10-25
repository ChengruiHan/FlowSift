"""Tests for episode grouping, broadcasting, clipping, and KL placement."""

import torch

from flowvla_rl.grpo import compute_group_advantages, grpo_loss, k3_kl_estimate


def test_group_advantage_centers_each_reset_group_without_std() -> None:
    rewards = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    advantages = compute_group_advantages(rewards)
    torch.testing.assert_close(
        advantages,
        torch.tensor([[-0.5, -0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]),
    )


def test_grpo_broadcasts_episode_advantage_over_chunks_and_denoise_steps() -> None:
    logp = torch.zeros(2, 3, 4, requires_grad=True)
    old = torch.zeros_like(logp)
    ref = torch.zeros_like(logp)
    advantages = torch.tensor([1.0, -1.0])
    loss, metrics = grpo_loss(
        logp, old, ref, advantages, clip_epsilon=0.2, kl_beta=0.0
    )
    assert loss.item() == 0.0
    loss.backward()
    assert logp.grad is not None
    torch.testing.assert_close(logp.grad[0], torch.full((3, 4), -1.0 / 24.0))
    torch.testing.assert_close(logp.grad[1], torch.full((3, 4), 1.0 / 24.0))
    assert metrics["ratio_mean"].item() == 1.0


def test_invalid_control_chunk_can_be_masked() -> None:
    logp = torch.zeros(1, 2, 2, requires_grad=True)
    valid = torch.tensor([[[True, True], [False, False]]])
    loss, _ = grpo_loss(
        logp,
        torch.zeros_like(logp),
        torch.zeros_like(logp),
        torch.ones(1),
        clip_epsilon=0.2,
        kl_beta=0.0,
        valid_steps=valid,
    )
    loss.backward()
    assert torch.count_nonzero(logp.grad[:, 1]) == 0


def test_k3_is_nonnegative_and_zero_at_reference() -> None:
    current = torch.tensor([-2.0, -1.0, 0.0])
    reference = torch.tensor([-1.0, -1.0, -1.0])
    estimate = k3_kl_estimate(current, reference)
    assert torch.all(estimate >= 0)
    assert estimate[1].item() == 0.0

