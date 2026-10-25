"""All-ten-step drift and sparse fixed-cache gradient diagnostics."""
from __future__ import annotations

import math

import torch

from flowvla_rl.checkpointing import capture_rng, restore_rng
from flowvla_rl.grpo import grpo_loss
from flowvla_rl.smolvla_sde import prepare_policy_prefix, recompute_log_probs, rollout_autocast
from flowvla_rl.step_selection import select_steps


@torch.no_grad()
def score_drift(policy, reference, chunks, attempted):
    device = next(policy.parameters()).device
    rng = capture_rng()
    mode = policy.training
    rows = []
    try:
        policy.eval()
        reference.eval()
        for stored in chunks:
            chunk = stored.to(device)
            with rollout_autocast(device):
                masks, cache = prepare_policy_prefix(policy, chunk.conditioning_batch)
                current = recompute_log_probs(policy.model, masks, cache, chunk.trajectory,
                                              valid_positions=chunk.valid_positions)
                ref = recompute_log_probs(reference, masks, cache, chunk.trajectory,
                                          valid_positions=chunk.valid_positions)
            delta = current - chunk.trajectory.masked_log_probs(chunk.valid_positions)
            d = ref - current
            k3 = d.exp() - d - 1
            for k in range(10):
                values = [float(delta[0, k]), float(delta[0, k].exp()), float(k3[0, k])]
                rows.append({"attempted_groups": attempted, "episode_id": chunk.episode_id,
                             "chunk_id": chunk.chunk_id, "step": k,
                             "delta_logp": values[0] if math.isfinite(values[0]) else None,
                             "ratio": values[1] if math.isfinite(values[1]) else None,
                             "k3": values[2] if math.isfinite(values[2]) else None,
                             "nonfinite_count": sum(not math.isfinite(x) for x in values)})
    finally:
        policy.train(mode)
        restore_rng(rng)
    return rows


def gradient_probe(policy, reference, chunks, root_seed):
    """One Full, one Late and eight Uniform gradients per representative chunk.

    Keep only one Full vector on CPU. No per-step gradient matrix is retained.
    """
    device = next(policy.parameters()).device
    rng, mode = capture_rng(), policy.training
    rows = []
    try:
        policy.eval()
        reference.eval()
        for chunk_index, stored in enumerate(chunks[:3]):
            chunk = stored.to(device)
            full_grads = None
            for method, draw in [("full10", 0), ("late5", 0)] + [("uniform5", i) for i in range(8)]:
                policy.zero_grad(set_to_none=True)
                steps = select_steps(method, root_seed=root_seed, attempted_group_id=draw,
                                     episode_id=chunk.episode_id, chunk_id=chunk.chunk_id)
                with rollout_autocast(device):
                    with torch.no_grad():
                        masks, cache = prepare_policy_prefix(policy, chunk.conditioning_batch)
                        ref = recompute_log_probs(reference, masks, cache, chunk.trajectory,
                                                  valid_positions=chunk.valid_positions, step_indices=steps)
                    current = recompute_log_probs(policy.model, masks, cache, chunk.trajectory,
                                                  valid_positions=chunk.valid_positions, step_indices=steps)
                old = chunk.trajectory.masked_log_probs(chunk.valid_positions)[:, steps]
                loss, _ = grpo_loss(current[:, None], old[:, None], ref[:, None],
                                    torch.tensor([chunk.advantage], device=device),
                                    clip_epsilon=0.2, kl_beta=0.01)
                loss.backward()
                if full_grads is None:
                    full_grads = {n: p.grad.detach().cpu().clone() for n, p in policy.named_parameters()
                                  if p.grad is not None}
                dot = norm2 = full2 = 0.0
                for name, p in policy.named_parameters():
                    if p.grad is not None:
                        g, f = p.grad.detach().float().cpu(), full_grads[name].float()
                        dot += float((g * f).sum())
                        norm2 += float(g.square().sum())
                        full2 += float(f.square().sum())
                cosine = dot / math.sqrt(norm2 * full2) if min(norm2, full2) > 1e-16 else None
                rows.append({"chunk_index": chunk_index, "method": method, "draw": draw,
                             "grad_norm": math.sqrt(norm2), "cosine_with_full": cosine})
    finally:
        policy.zero_grad(set_to_none=True)
        policy.train(mode)
        restore_rng(rng)
    return rows
