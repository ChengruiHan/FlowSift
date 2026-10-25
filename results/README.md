# Generated experiment artifacts

This directory stores generated run data. Large artifacts are excluded from Git by default; curated historical evidence is published under [docs/experiments/seed101](../docs/experiments/seed101/README.md).

- `cpu-smoke/`, `cpu-profile/`, `cpu-correctness.json`: toy CPU validation, not real SmolVLA results.
- `server_environment.json`, `server-requirements.lock.txt`: the actual server environment.
- `profile_caches/`: Stage-A trajectories and pre-update snapshots.
- `runs/`: pilot/formal manifests, groups, failures, sessions and checkpoints.
- `evaluations/`: per-episode ODE evaluations for each checkpoint.
- `report/`: per-seed/per-task results, paired cache costs and figures.

Keep original logs, failures and every planned seed. Missing measurements remain missing; never fill them with invented example data or encode a failed run as zero success.
