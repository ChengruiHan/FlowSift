"""Memory-conscious frozen reference construction for SmolVLA KL."""

from __future__ import annotations

import copy

from torch import nn


def make_frozen_reference(model: nn.Module) -> nn.Module:
    """Copy the action expert while sharing the already-frozen VLM.

    SmolVLA's observation prefix cache depends only on the VLM and frozen state
    path, so policy and reference can also consume the same cache. This avoids a
    third policy (old log-probs are cached during rollout) and avoids duplicating
    the ~350M-parameter VLM.
    """
    vlm = model.vlm_with_expert.vlm
    if any(parameter.requires_grad for parameter in vlm.parameters()):
        raise ValueError("the VLM must be frozen before constructing the reference")
    # Do not transiently duplicate the frozen VLM on GPU during deepcopy.
    reference = copy.deepcopy(model, memo={id(vlm): vlm})
    reference.vlm_with_expert.vlm = vlm
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference.eval()
    return reference


def shared_vlm_parameter_count(model: nn.Module, reference: nn.Module) -> int:
    """Assert identity sharing and return the shared VLM parameter count."""
    policy_parameters = list(model.vlm_with_expert.vlm.parameters())
    reference_parameters = list(reference.vlm_with_expert.vlm.parameters())
    if len(policy_parameters) != len(reference_parameters) or any(
        left is not right
        for left, right in zip(policy_parameters, reference_parameters, strict=True)
    ):
        raise AssertionError("policy and reference do not share the same VLM tensors")
    return sum(parameter.numel() for parameter in policy_parameters)

