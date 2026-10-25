# Recorded single-seed experiment

This directory contains evidence from the completed 2026-10-22/23 LIBERO-Object experiment: author SFT; formal seed 101; 300 attempted groups per method; pilots at seed 17 with 100 groups.

- `evaluations/`: per-episode ODE records, manifests and summaries (test, dev and replay).
- `runs/`: six pilot/formal manifests, summaries, group logs and session records.
- `gpu_profile/`: repeated fixed-cache timings, phase timings and gradient probes.
- `report/`: original tables and plots.
- `gpu_correctness.json`: original passed GPU regression gate.
- `protocol.lock.json`: archived decision, with `test_opened=false` representing the state **when the lock was written**, not current evaluation status.
- `provenance.json`: original and public file hashes and original server source identity.
- `training-source.zip`: exact tracked source/config/test snapshot from the recorded training commit, including its original MIT license. This is the historical implementation, before the fixes on the public main branch.

Server commit: `de2095ecddf4b45f0a57c906c0950d7200ba6773` (identifier from the private server history, not a commit in this repository). The snapshot reproduces recorded source hash `c574041612ca139765d44bce3946ee77ce8a6d10774c2b419a505d0095b5203e`.

Absolute server project paths in published text artifacts are replaced with `<PROJECT>`. Numbers, policy hashes and source hashes are unchanged. `provenance.json` lists which files changed and both hashes. Therefore the redacted lock is **not a reusable protocol approval**; metadata hashes referring to the original lock describe the original unredacted artifact. Weights, optimizer states, caches and private deployment history are not published.

From the project root:

```bash
python scripts/verify_published_results.py
python scripts/summarize_results.py \
  --runs docs/experiments/seed101/runs \
  --evaluations docs/experiments/seed101/evaluations \
  --profile docs/experiments/seed101/gpu_profile/learner_profile.csv \
  --output results/reproduced-report
```

The first command checks every published evidence-file hash, the archived source hash, evaluation-to-final-checkpoint identity, success rates and training costs. It does not rerun training. For a new experiment, use the [training guide](../../server_training.md) and fresh output directories, gates and lock.

See [limitations](../../results_and_limitations.md) and [post-experiment review](../../code_review.md) before interpreting these observations. Multiple evaluation episodes do not replace independent training seeds.
