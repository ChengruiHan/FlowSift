# Correctness evidence

The reviewed implementation passed **112 CPU tests** on 2026-10-25. The [review](code_review.md) documents the added boundary fixes and [Linux CI](https://github.com/ChengruiHan/FlowSift/actions/runs/40154856010) passed. The completed GPU experiment has separate [historical evidence](experiments/seed101/README.md); its gate must not be reused as approval for modified source.

## Original CPU validation

On 2026-10-22, the initial implementation passed **95 tests**, including inherited upstream regressions, on macOS ARM64 / Python 3.12.14 / PyTorch 2.14.1 CPU. Additional commands exercised integration, cache verification, profiling and syntax:

```bash
python -m pytest -q
python scripts/smoke_cpu.py --output results/cpu-smoke
python scripts/verify_cache.py --toy --output results/cpu-correctness.json
python scripts/profile_learner.py --toy --output results/cpu-profile --gradient-probe
python scripts/train_grpo_libero.py --config configs/uniform5.yaml --validate-only
python -m compileall -q src scripts
bash -n scripts/setup_server.sh
```

Each CPU integration run completed three attempted groups, two updates and a collapsed final group, and saved a final checkpoint. CPU profiling ran 3 caches × 3 methods × 3 repetitions (27 measurements). Measured current/reference calls were 10/10 for Full and 5/5 for Late/Uniform. Toy timings are not real-model speed or memory results.

## Gate coverage

| Gate | CPU evidence | Real-runtime scope and limitations |
|---|---|---|
| C1 | Unmodified upstream Full, omitted indices and all-index paths agree on log-probabilities, loss, gradients and updates; explicit state-projection freezing does not change actual updates. | The archived GPU gate compares Full rescoring and one update against the upstream snapshot. |
| C2 | Arbitrarily ordered subsets equal corresponding full slices, retaining original times and masks. | Included in archived GPU cache verification. |
| C3 | Instrumented denoiser calls are 10/10 or 5/5 per chunk. | Recorded GPU profile also measures call counts. |
| C4 | Nonempty, unique, in-range selections; constant objectives do not shrink when fewer steps are selected. | Same selection/normalization implementation is used on GPU. |
| C5 | Enumerating all 252 subsets preserves mean loss and per-parameter gradients, including nonzero k3 and nonunit ratios. | Real-model probes use three chunks and eight Uniform samples per chunk, not exhaustive enumeration or a numerical proof of unbiasedness. |
| C6 | Density aggregation respects masks; cached paths and reference parameters have no gradients. | CPU fixtures do not exercise every real preprocessing boundary. |
| C7 | Ten-step old/current agreement; corrupting a later chunk aborts without an optimizer step. | Online selected-transition checks require absolute log-probability error ≤0.01 and finite values. |
| C8 | Actual gradient names match the declared set; initial clipped/unclipped PG gradients agree. | Archived GPU gate records actual gradient parameter names. |
| C9 | Selection and diagnostics preserve Python, NumPy and torch RNG; stable keys reproduce indices. | Exact simulator reconstruction after interruption is not established. |
| C10 | All-collapsed, final-collapsed, mixed, partial-group and already-at-budget cases terminate/save correctly. | Live GPU failure injection was not performed. |
| C11 | Two continuous updates equal an interrupted CPU save/restore sequence; RNG and reference hash are preserved; metadata mismatches are rejected. | GPU optimizer restoration and subsequent simulator sampling were not exercised by a real interruption. |

Other tests cover protocol rejection, budget calculation, method configurations, backend-separated cache comparisons, duplicate/missing evaluation episodes, and the review's new provenance and early-stop cases.

CPU floating comparisons use `atol=1e-6, rtol=1e-5`; checkpoint continuation comparisons use zero tolerance. `verify_cache.py` records maximum absolute and relative L2 errors for log-probabilities and updates. Real BF16 tolerances must be justified consistently from baseline behavior, not relaxed per method.

## Historical GPU evidence

The recorded GPU cache gate passed. SFT replay checks agreed on ten sampled conditions. CUDA, model-file hashes, LIBERO/EGL, online training and final ODE evaluation were exercised in the completed single-seed workflow.

This does **not** mean every C1–C11 property or every recovery failure path was exhaustively validated on the real model. Consult the raw [GPU gate](experiments/seed101/gpu_correctness.json) and [review limitations](code_review.md) for the distinction.
