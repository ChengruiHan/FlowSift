"""CPU regression evidence for C1-C11. GPU and simulator checks remain separate."""
import copy
import importlib.util
import itertools
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from flowvla_rl import training
from flowvla_rl.checkpointing import (RunState, capture_rng, read_checkpoint, restore_checkpoint,
                                     restore_rng, save_checkpoint, state_hash)
from flowvla_rl.diagnostics import score_drift
from flowvla_rl.grpo import grpo_loss
from flowvla_rl.protocol import planned_budget
from flowvla_rl.reference import make_frozen_reference
from flowvla_rl.runner import run_budget
from flowvla_rl.sde import gaussian_log_prob
from flowvla_rl.smolvla_sde import prepare_policy_prefix, recompute_log_probs
from flowvla_rl.step_selection import select_steps, selection_seed, validate_indices
from flowvla_rl.toy_backend import TinyPolicy, make_chunks


def upstream(name):
    spec = importlib.util.spec_from_file_location("original_" + name, Path(__file__).parent / "upstream" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup():
    torch.manual_seed(123)
    policy = TinyPolicy()
    ref = make_frozen_reference(policy.model)
    return policy, ref, make_chunks(policy)


def optimizer(policy):
    return torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-6, weight_decay=1e-10)


def update(policy, ref, chunks, opt, **kwargs):
    return training.optimize_grpo(policy, ref, chunks, opt, clip_epsilon=0.2, kl_beta=0.01,
                                  epochs=1, max_grad_norm=1, seed=12, **kwargs)


def score(policy, chunk, indices=None):
    with torch.no_grad():
        masks, cache = prepare_policy_prefix(policy, chunk.conditioning_batch)
    return recompute_log_probs(policy.model, masks, cache, chunk.trajectory,
                               valid_positions=chunk.valid_positions, step_indices=indices)


def objective(policy, reference, chunk, indices=None):
    steps = tuple(range(10)) if indices is None else indices
    logp = score(policy, chunk, steps)
    with torch.no_grad():
        masks, cache = prepare_policy_prefix(policy, chunk.conditioning_batch)
        ref = recompute_log_probs(reference, masks, cache, chunk.trajectory,
                                 valid_positions=chunk.valid_positions, step_indices=steps)
    old = chunk.trajectory.masked_log_probs(chunk.valid_positions)[:, steps]
    return grpo_loss(logp[:, None], old[:, None], ref[:, None], torch.tensor([chunk.advantage]),
                      clip_epsilon=0.2, kl_beta=0.01)[0]


def test_c1_upstream_full_logp_loss_gradients_and_update(setup):
    policy, ref, chunks = setup
    original = upstream("smolvla_sde")
    chunk = chunks[0]
    with torch.no_grad():
        masks, cache = prepare_policy_prefix(policy, chunk.conditioning_batch)
    expected = original.recompute_log_probs(policy.model, masks, cache, chunk.trajectory,
                                            valid_positions=chunk.valid_positions)
    for indices in (None, tuple(range(10))):
        actual = score(policy, chunk, indices)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        params = [p for p in policy.parameters() if p.requires_grad]
        left = torch.autograd.grad(actual.mean(), params)
        right = torch.autograd.grad(expected.mean(), params, retain_graph=True)
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    current, old = copy.deepcopy(policy), copy.deepcopy(policy)
    # state_proj was declared trainable upstream, but receives no gradients under no_grad prefix.
    old.model.state_proj.requires_grad_(True)
    old_ref, current_ref = make_frozen_reference(old.model), make_frozen_reference(current.model)
    original_learner = upstream("training")
    old_metrics = original_learner.optimize_grpo(old, old_ref, chunks, optimizer(old),
                           clip_epsilon=0.2, kl_beta=0.01, epochs=1, max_grad_norm=1, seed=12)
    new_metrics = update(current, current_ref, chunks, optimizer(current))
    assert old_metrics["loss"] == pytest.approx(new_metrics["loss"], abs=1e-6)
    for name, value in current.state_dict().items():
        torch.testing.assert_close(value, old.state_dict()[name], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("indices", [(5, 6, 7, 8, 9), (9, 0, 4, 2, 7), (0,), tuple(range(10))])
def test_c2_subset_preserves_original_time_grid_and_mask(setup, indices):
    policy, _, chunks = setup
    with patch.object(policy.model, "denoise_step", wraps=policy.model.denoise_step) as calls:
        actual = score(policy, chunks[-1], indices)
        assert calls.call_count == len(indices)
        assert [float(c.kwargs["timestep"][0]) for c in calls.call_args_list] == pytest.approx([1-k/10 for k in indices])
    torch.testing.assert_close(actual, score(policy, chunks[-1])[:, indices])


@pytest.mark.parametrize("method,count", [("full10", 10), ("late5", 5), ("uniform5", 5)])
def test_c3_actual_current_reference_calls(setup, method, count):
    policy, ref, chunks = setup
    with patch.object(policy.model, "denoise_step", wraps=policy.model.denoise_step) as a, \
         patch.object(ref, "denoise_step", wraps=ref.denoise_step) as b:
        metrics = update(policy, ref, chunks, optimizer(policy), method=method)
        assert a.call_count == b.call_count == count * len(chunks)
        assert metrics["current_calls"] == a.call_count
    assert all(c.trajectory.states.shape[1] == 11 for c in chunks)


@pytest.mark.parametrize("indices", [[], [1, 1], [-1], [10], [1.5], [True]])
def test_c4_invalid_indices(indices):
    with pytest.raises(ValueError):
        validate_indices(indices, 10)


def test_c4_constant_objective_normalization():
    for m in (5, 10):
        logp = torch.zeros(1, 1, m, requires_grad=True)
        loss, _ = grpo_loss(logp, logp.detach(), logp.detach(), torch.tensor([0.5]),
                            clip_epsilon=0.2, kl_beta=0.01)
        assert loss.item() == -0.5


def test_c4_c5_all_252_subset_losses_and_gradients(setup):
    policy, ref, chunks = setup
    # Nonzero KL / non-unit ratios: verifies the whole combination, not just the initial PG term.
    with torch.no_grad():
        policy.model.action_out_proj.bias.add_(0.002)
    params = [p for p in policy.parameters() if p.requires_grad]
    full_loss = objective(policy, ref, chunks[0])
    full_grad = torch.autograd.grad(full_loss, params)
    summed = [torch.zeros_like(g) for g in full_grad]
    losses = []
    subsets = list(itertools.combinations(range(10), 5))
    assert len(subsets) == 252
    for subset in subsets:
        loss = objective(policy, ref, chunks[0], subset)
        losses.append(float(loss.detach()))
        for total, grad in zip(summed, torch.autograd.grad(loss, params)):
            total.add_(grad / 252)
    assert sum(losses) / 252 == pytest.approx(float(full_loss.detach()), abs=1e-6)
    for actual, expected in zip(summed, full_grad):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_c6_density_mask_and_cache_detach(setup):
    mean = torch.zeros(1, 2, 3, requires_grad=True)
    x = torch.zeros_like(mean)
    mask = torch.tensor([[[True, True, False], [False, False, False]]])
    a = gaussian_log_prob(x, mean, torch.tensor(0.1), mask=mask)
    changed = x.clone()
    changed[~mask] = 1000
    b = gaussian_log_prob(changed, mean, torch.tensor(0.1), mask=mask)
    torch.testing.assert_close(a, b)
    policy, ref, chunks = setup
    chunks[0].trajectory.states.requires_grad_()
    chunks[0].trajectory.element_log_probs.requires_grad_()
    update(policy, ref, chunks, optimizer(policy))
    assert chunks[0].trajectory.states.grad is None
    assert chunks[0].trajectory.element_log_probs.grad is None
    assert all(p.grad is None for p in ref.parameters())


def test_c7_all_steps_match_before_update_and_bad_later_chunk_fails(setup):
    policy, ref, chunks = setup
    for chunk in chunks:
        delta = score(policy, chunk).detach() - chunk.trajectory.masked_log_probs(chunk.valid_positions)
        assert torch.isfinite(delta).all() and float(delta.abs().max()) <= 1e-6
    chunks[-1].trajectory.element_log_probs[:, 7, 0, 0] += 0.02
    before = state_hash(policy)
    with pytest.raises(RuntimeError, match="old/current mismatch"):
        update(policy, ref, chunks, optimizer(policy))
    assert state_hash(policy) == before


def test_c8_gradient_scope_and_initial_clip_equivalence(setup):
    policy, ref, chunks = setup
    metrics = update(policy, ref, chunks, optimizer(policy))
    assert set(metrics["actual_gradient_names"]) == {n for n, p in policy.named_parameters() if p.requires_grad}
    assert not any("state_proj" in n or ".vlm." in n for n in metrics["actual_gradient_names"])
    logp = torch.randn(1, 1, 10, requires_grad=True)
    old = logp.detach().clone()
    clipped, _ = grpo_loss(logp, old, old, torch.tensor([0.5]), clip_epsilon=0.2, kl_beta=0)
    unclipped = -((logp-old).exp() * 0.5).mean()
    torch.testing.assert_close(torch.autograd.grad(clipped, logp)[0], torch.autograd.grad(unclipped, logp)[0])


def test_c9_local_selection_and_diagnostics_do_not_advance_rng(setup):
    policy, ref, chunks = setup
    before = capture_rng()
    args = dict(root_seed=selection_seed(101), attempted_group_id=5, episode_id=20, chunk_id=2)
    first = select_steps("uniform5", **args)
    for _ in range(100):
        assert select_steps("uniform5", **args) == first
    assert len(first) == len(set(first)) == 5
    score_drift(policy, ref, chunks[:1], 50)
    after = capture_rng()
    assert before["python"] == after["python"]
    assert np.array_equal(before["numpy"][1], after["numpy"][1])
    assert torch.equal(before["torch"], after["torch"])


@pytest.mark.parametrize("mixed", [[False]*3, [True, True, False], [True, False, True]])
def test_c10_attempted_budget_and_final(mixed):
    saves, learns, logs = [], [], []
    def collect(i, start):
        return [SimpleNamespace(success=mixed[i] and j == 0, steps=7) for j in range(4)], [], {}
    result = run_budget(budget=3, state=RunState(), collect=collect,
                        learn=lambda c, i: learns.append(i) or {},
                        save=lambda s, final: saves.append((s.attempted_groups, final)) or "checkpoint",
                        log=logs.append, save_every=2)
    assert result["attempted_groups"] == 3 and result["episodes"] == 12 and result["env_steps"] == 84
    assert result["completed_updates"] == sum(mixed)
    assert saves[-1] == (3, True)
    resumed = RunState(attempted_groups=3)
    run_budget(budget=3, state=resumed, collect=lambda *a: pytest.fail("collected past budget"),
               learn=None, save=lambda s, final: saves.append((s.attempted_groups, final)), log=lambda r: None)
    assert saves[-1] == (3, True)


def test_c10_partial_group_does_not_count():
    state = RunState()
    with pytest.raises(ValueError, match="complete"):
        run_budget(budget=1, state=state, collect=lambda *a: ([SimpleNamespace(success=False, steps=1)], [], {}),
                   learn=None, save=None, log=None)
    assert state.attempted_groups == state.episodes == 0


def test_c11_checkpoint_restores_current_optimizer_rng_and_sft_reference(setup, tmp_path):
    policy, ref, chunks = setup
    initial = copy.deepcopy(policy)
    opt = optimizer(policy)
    update(policy, ref, chunks, opt, method="uniform5")
    metadata = {"reference_hash": state_hash(ref), "method": "uniform5", "seed": 101}
    saved_path = save_checkpoint(tmp_path, policy, opt, RunState(attempted_groups=1), metadata)
    expected_random = (random.random(), np.random.rand(), torch.rand(2))
    next_chunks = make_chunks(policy, seed=20)
    update(policy, ref, next_chunks, opt, method="uniform5", attempted_group_id=1)
    restored = initial
    restored_ref = make_frozen_reference(restored.model)
    restored_opt = optimizer(restored)
    saved = read_checkpoint(tmp_path / "latest.json")
    restored_state = restore_checkpoint(saved, restored, restored_opt, metadata)
    assert restored_state.attempted_groups == 1
    assert random.random() == expected_random[0] and np.random.rand() == expected_random[1]
    assert torch.equal(torch.rand(2), expected_random[2])
    update(restored, restored_ref, next_chunks, restored_opt, method="uniform5", attempted_group_id=1)
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    assert state_hash(restored_ref) == metadata["reference_hash"] == state_hash(ref)
    assert state_hash(restored.model) != state_hash(restored_ref)
    with pytest.raises(ValueError, match="mismatch"):
        restore_checkpoint(saved, restored, restored_opt, {**metadata, "method": "late5"})


@pytest.mark.parametrize("mixed,budget", [(80,300), (60,400), (40,500), (20,600)])
def test_budget_rule(mixed, budget):
    assert planned_budget(mixed) == budget


def test_resumed_loop_retains_committed_checkpoint_cost(monkeypatch):
    from flowvla_rl import runner
    clock = [0.0]
    monkeypatch.setattr(runner.time, 'perf_counter', lambda: clock[0])
    logs, payloads = [], []

    def collect(i, start):
        if i == 1:
            raise RuntimeError('simulated interruption after checkpoint')
        clock[0] += 10
        return [SimpleNamespace(success=False, steps=1) for _ in range(4)], [], {}

    def save(state, final):
        payloads.append(copy.deepcopy(state))
        clock[0] += 3
        return 'checkpoint'

    with pytest.raises(RuntimeError, match='interruption'):
        run_budget(budget=2, state=RunState(), collect=collect, learn=None,
                   save=save, log=logs.append, save_every=1)
    restored = payloads[-1]
    assert restored.elapsed_seconds == 10
    runner.restore_elapsed_from_log(restored, logs)
    assert restored.elapsed_seconds == 13
    result = run_budget(budget=1, state=restored, collect=None, learn=None,
                        save=save, log=logs.append)
    assert result['elapsed_seconds'] == 16
    measured = sum(row.get('rollout_seconds', 0) + row.get('checkpoint_seconds', 0) for row in logs)
    assert result['elapsed_seconds'] == measured
