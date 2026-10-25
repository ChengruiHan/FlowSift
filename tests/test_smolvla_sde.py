"""Interface tests using a tiny stand-in for VLAFlowMatching."""

from types import SimpleNamespace

import torch
from torch import nn

from flowvla_rl.smolvla_sde import recompute_log_probs, sample_sde_chunk


class TinyFlowModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.config = SimpleNamespace(chunk_size=4, max_action_dim=5, num_steps=3)

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def denoise_step(self, *, x_t, prefix_pad_masks, past_key_values, timestep):
        del prefix_pad_masks, past_key_values
        return self.scale * x_t + timestep[:, None, None]


def test_sample_and_rescore_match_before_update() -> None:
    torch.manual_seed(0)
    model = TinyFlowModel()
    prefix_mask = torch.ones(2, 3, dtype=torch.bool)
    trajectory = sample_sde_chunk(
        model, prefix_mask, None, action_dim=3, eta=0.4
    )
    assert trajectory.states.shape == (2, 4, 4, 5)
    assert trajectory.element_log_probs.shape == (2, 3, 4, 5)
    assert trajectory.log_probs.shape == (2, 3)
    rescored = recompute_log_probs(model, prefix_mask, None, trajectory)
    torch.testing.assert_close(rescored.detach(), trajectory.log_probs)
    rescored.mean().backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)


def test_unexecuted_chunk_positions_can_be_excluded() -> None:
    torch.manual_seed(1)
    model = TinyFlowModel()
    prefix_mask = torch.ones(1, 2, dtype=torch.bool)
    trajectory = sample_sde_chunk(
        model, prefix_mask, None, action_dim=3, eta=0.4
    )
    valid_positions = torch.tensor([[True, True, False, False]])
    rescored = recompute_log_probs(
        model,
        prefix_mask,
        None,
        trajectory,
        valid_positions=valid_positions,
    )
    assert rescored.shape == (1, 3)
    assert torch.isfinite(rescored).all()
    torch.testing.assert_close(
        rescored.detach(), trajectory.masked_log_probs(valid_positions)
    )
