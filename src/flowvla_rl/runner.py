"""Attempted-group budget loop, independent of the heavy simulator runtime."""
from __future__ import annotations

import time
from dataclasses import asdict

from flowvla_rl.checkpointing import RunState


def restore_elapsed_from_log(state: RunState, rows):
    """Recover committed checkpoint time not yet present in its own payload.

    Call only with rows retained at the restored complete-group boundary.
    Interrupted/replayed work belongs in the separate resource ledger.
    """
    state.elapsed_seconds = max(
        [state.elapsed_seconds] + [float(row["elapsed_seconds"]) for row in rows
                                  if "elapsed_seconds" in row
                                  and row.get("attempted_groups", -1) <= state.attempted_groups]
    )


def run_budget(*, budget: int, state: RunState, collect, learn, save, log,
               diagnose=None, save_every=50, diagnostic_every=50):
    if budget < state.attempted_groups or budget <= 0:
        raise ValueError("budget cannot precede restored attempted count")
    started, previous_elapsed = time.perf_counter(), state.elapsed_seconds
    if diagnose is not None and state.attempted_groups == 0:
        begin = time.perf_counter()
        diagnose(0)
        log({"event": "diagnostic", "attempted_groups": 0,
             "diagnostic_seconds": time.perf_counter() - begin})
    while state.attempted_groups < budget:
        attempt_id = state.attempted_groups
        begin = time.perf_counter()
        episodes, chunks, context = collect(attempt_id, state.episodes)
        rollout_seconds = time.perf_counter() - begin
        if len(episodes) != 4:
            raise ValueError("only complete G=4 groups may consume the attempted budget")
        rewards = [int(e.success) for e in episodes]
        mixed = 0 < sum(rewards) < 4
        # Update only after the complete group passes collection/identity checks.
        metrics = learn(chunks, attempt_id) if mixed else {"learner_seconds": 0.0}
        state.attempted_groups += 1
        state.episodes += len(episodes)
        state.env_steps += sum(e.steps for e in episodes)
        state.completed_updates += int(mixed)
        state.collapsed_groups += int(not mixed)
        diagnostic_seconds = 0.0
        if diagnose is not None and state.attempted_groups % diagnostic_every == 0:
            begin = time.perf_counter()
            diagnose(state.attempted_groups)
            diagnostic_seconds = time.perf_counter() - begin
        state.elapsed_seconds = previous_elapsed + time.perf_counter() - started
        row = {**context, **metrics, "attempted_id": attempt_id,
               "attempted_groups": state.attempted_groups, "rewards": rewards, "mixed": mixed,
               "episode_count": len(episodes), "env_steps": sum(e.steps for e in episodes),
               "completed_updates": state.completed_updates, "chunk_count": len(chunks),
               "rollout_seconds": rollout_seconds, "diagnostic_seconds": diagnostic_seconds,
               "checkpoint_seconds": 0.0, "elapsed_seconds": state.elapsed_seconds}
        # Commit the row before checkpointing, so a saved boundary always has its group log.
        log(row)
        if state.attempted_groups % save_every == 0:
            begin = time.perf_counter()
            save(state, False)
            log({"event": "checkpoint", "attempted_groups": state.attempted_groups,
                 "checkpoint_seconds": time.perf_counter() - begin,
                 "elapsed_seconds": previous_elapsed + time.perf_counter() - started})
    if diagnose is not None and (state.attempted_groups == 0 or state.attempted_groups % diagnostic_every):
        begin = time.perf_counter()
        diagnose(state.attempted_groups)
        log({"event": "diagnostic", "attempted_groups": state.attempted_groups,
             "diagnostic_seconds": time.perf_counter() - begin})
    state.elapsed_seconds = previous_elapsed + time.perf_counter() - started
    begin = time.perf_counter()
    final_path = save(state, True)
    log({"event": "checkpoint", "attempted_groups": state.attempted_groups,
         "checkpoint_seconds": time.perf_counter() - begin, "final": True,
         "elapsed_seconds": previous_elapsed + time.perf_counter() - started})
    state.elapsed_seconds = previous_elapsed + time.perf_counter() - started
    return {**asdict(state), "status": "complete", "final_checkpoint": str(final_path),
            "collapse_rate": state.collapsed_groups / max(state.attempted_groups, 1)}
