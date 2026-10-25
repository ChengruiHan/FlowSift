"""Reference policy must share only the frozen backbone, never expert weights."""

from types import SimpleNamespace

import torch
from torch import nn

from flowvla_rl.reference import make_frozen_reference, shared_vlm_parameter_count


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vlm_with_expert = nn.Module()
        self.vlm_with_expert.vlm = nn.Linear(3, 3)
        self.vlm_with_expert.lm_expert = nn.Linear(3, 2)
        self.action_out_proj = nn.Linear(2, 1)
        for parameter in self.vlm_with_expert.vlm.parameters():
            parameter.requires_grad_(False)
        self.config = SimpleNamespace()


def test_reference_shares_vlm_but_copies_expert() -> None:
    model = TinyModel()
    reference = make_frozen_reference(model)
    expected = sum(parameter.numel() for parameter in model.vlm_with_expert.vlm.parameters())
    assert shared_vlm_parameter_count(model, reference) == expected
    assert (
        model.vlm_with_expert.lm_expert.weight
        is not reference.vlm_with_expert.lm_expert.weight
    )
    assert not any(parameter.requires_grad for parameter in reference.parameters())
    with torch.no_grad():
        model.vlm_with_expert.lm_expert.weight.add_(1.0)
    assert not torch.equal(
        model.vlm_with_expert.lm_expert.weight,
        reference.vlm_with_expert.lm_expert.weight,
    )

