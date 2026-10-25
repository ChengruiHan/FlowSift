# Results and limitations

FlowSift measures whether reducing learner transition rescoring produces a useful cost–performance trade-off while preserving the complete action-generation procedure. The recorded experiment completed on **2026-10-23 at 03:44:09, Asia/Shanghai**.

All public measurements, per-episode records and the historical source snapshot are in the [seed-101 evidence directory](experiments/seed101/README.md). The later [code review](code_review.md) distinguishes that implementation from the reviewed main branch.

## Experimental setting

| Setting | Recorded value |
|---|---|
| SFT source | `MorpheusTzz/smolvla-grpo-libero-object/sft-100pct-baseline` |
| Model revision | `a120aa9b51f8e55db5ad8284e051963c61a2f5ed` |
| Hardware | RTX 4090 D, 24 GB |
| Runtime | PyTorch 2.8.0+cu128; LeRobot 0.6.0 |
| Pilots | Three methods, seed 17, 100 attempted groups each |
| Formal budget | Three methods, seed 101, 300 attempted groups each |
| Sampling budget | G=4; 1,200 training episodes per formal method |
| Final evaluation | Final checkpoint; 10 LIBERO-Object tasks × 20 episodes |
| Evaluation sampler | 10-step ODE; maximum 280 control steps per episode |

Each method starts independently from the same SFT. The Full-10 pilot selects the common formal budget. The workflow includes replay checks, cache collection, a GPU regression gate, fixed-cache profiling, pilots, development evaluation, budget locking, formal training and final evaluation. Total workflow wall time was approximately **9 hours 49 minutes**, which is not the complete billed allocation time.

## Final task performance

| Model | Successes | Success rate | Change vs SFT | Change vs Full-10 | Effective updates | Training loop |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 120/200 | 60.0% | — | — | — | — |
| Full-10 | 127/200 | 63.5% | +3.5 pp | — | 208 | 105.9 min |
| Late-5 | 133/200 | 66.5% | +6.5 pp | +3.0 pp | 201 | 91.7 min |
| Uniform-5 | 138/200 | 69.0% | +9.0 pp | +5.5 pp | 203 | 88.8 min |

Every model has exactly 200 test records and 20 records per task, with unique task/episode indices. Success counts agree with the summaries, and evaluated model hashes match the recorded final policies. All three methods reach the same attempted-group budget. Recorded failed-session time and replayed-group count are zero.

These are observed success rates for one trained checkpoint per method. The 200 evaluation episodes do not constitute 200 independent training repetitions.

## Where the time went

| Method | Rollout | Learner | Diagnostics | Checkpoints | Other | Total |
|---|---:|---:|---:|---:|---:|---:|
| Full-10 | 72.37 min | 32.96 min | 0.28 min | 0.25 min | 0.07 min | 105.93 min |
| Late-5 | 73.59 min | 17.54 min | 0.29 min | 0.25 min | 0.04 min | 91.71 min |
| Uniform-5 | 71.88 min | 16.34 min | 0.26 min | 0.23 min | 0.04 min | 88.76 min |

Values are rounded; exact seconds are in the [cost table](experiments/seed101/report/system_cost.csv). The training loop includes rollout, learning, diagnostics and checkpointing. Model initialization and final evaluation are outside this timer.

Relative to Full-10, loop time is **13.4% lower for Late-5** and **16.2% lower for Uniform-5**. Rollout remains the largest cost. The online methods also collect different trajectories and perform different numbers of effective updates, so these loop measurements are distinct from the controlled cache benchmark.

## Fixed-cache GPU profile

The profile uses **3 cached groups × 3 methods × 3 repetitions**. Each comparison restores the same stored trajectories, current parameters and optimizer state. Ratios use the median of repetitions within each cache.

| Method | Learner time reduction vs Full | Speedup | Current / reference calls per chunk |
|---|---:|---:|---:|
| Full-10 | — | 1.00× | 10 / 10 |
| Late-5 | 46.9–48.0% | 1.88–1.92× | 5 / 5 |
| Uniform-5 | 36.3–50.8% | 1.57–2.03× | 5 / 5 |

The ranges describe the three cache comparisons, not confidence intervals. This benchmark isolates learner work; it does not establish policy quality. CUDA allocated/reserved peaks are process-level measurements, not total board usage. Phase timings and gradient probes are separate passes and are not added to natural-throughput measurements.

[Paired profile table](experiments/seed101/report/profile_paired.csv) · [Raw profile records](experiments/seed101/gpu_profile/learner_profile.csv)

## What the experiment establishes

In this SFT/seed/budget setting, both five-step implementations reduce measured learner cost and total loop time. Their observed final ODE success rates exceed Full-10's. Uniform-5 has the highest observed success rate in this run.

The study supplies three complementary forms of evidence: implementation invariants, compute comparisons on identical caches, and task evaluations under equal attempted-group budgets. None alone answers all three questions.

## Limitations

- **One formal training seed.** No claim of statistical significance, universal superiority or non-inferiority is supported.
- **One initialization and suite.** The official SFT and other LIBERO suites were not evaluated here. Results from different published protocols are not direct baselines.
- **No strictly held-out initial-state split.** Training, development and test evaluations use the LIBERO initial-state distribution. Strict unseen-scene generalization is not claimed.
- **Limited replay checks.** Two SFT checks agreed on all ten sampled conditions; this does not establish strict pairing for every evaluation. No paired bootstrap is reported.
- **No live interruption/recovery experiment.** GPU optimizer recovery and simulator reconstruction were not tested through an actual interrupted training run.
- **Historical implementation.** The recorded GPU gate passed for the archived source. Later early-termination and validation fixes have CPU coverage, but were not rerun on GPU. Old logs do not contain every step's termination reason, so they cannot completely rule out the early-stop edge case discussed in the audit.
- **Different measures of efficiency.** Equal episode/group budgets do not mean equal environment steps; learner acceleration does not imply equal inference acceleration or billing reductions.

## Evidence access and archival record

The [public evidence guide](experiments/seed101/README.md) links tables, raw evaluations, logs, protocol metadata and a source snapshot. Verify it with:

```bash
python scripts/verify_published_results.py
```

The complete local backup additionally contains weights, optimizer states, caches, initial SFT, source history and environment logs. On 2026-10-23 at 13:10:31 (Asia/Shanghai), **206 files totaling 20,689,756,462 bytes** passed individual SHA256 verification. The subsequent server sync/shutdown command returned zero. These private operational artifacts live under `results/server-experiment/` and are not distributed in Git.
