# Core code review — 2026-10-25

This review traced the v3 protocol through SDE sampling, transition selection, GRPO, the reference model, rollout collection, budgeting, checkpoints, evaluation and reporting. Focused regression tests were added for the edge cases below.

**Within the reviewed paths and test coverage, no implementation error was found in the selected indices, five-step normalization, shared-parameter updates or immutable SFT reference.** This is not a proof that every CUDA or simulator failure path is correct.

## Findings and fixes

| Priority | Trigger and previous behavior | Correction | Implication for recorded results |
|---|---|---|---|
| P1 | An unsuccessful early termination did not stop rollout. Truncation stopped only the inner chunk loop, so collection could continue. | End the whole episode on success, termination or truncation; preserve the executed-action mask. | The ordinary 280-step horizon was already capped. Old logs lack per-step termination reasons, so early-stop cases cannot be completely ruled out. Historical results were not rewritten. |
| P2 | A complete formal matrix did not prove that the selected run-id belonged to it. Reporting joined results by run-id without cross-checking policy hashes. | Check selected-run membership, CSV/manifest/summary consistency, final policy hash, evaluation seeds and episode indices. | All historical evaluations pass the stricter validation; reported success rates and costs are unchanged. |
| P2 | A checkpoint payload recorded elapsed time before its own save, while retained logs included the checkpoint cost. | Log cumulative time after saving and recover it from retained boundary records. | The historical runs did not resume; their costs are unchanged. |
| P2 | NaN/Inf observations could pass tolerance comparisons; identical digests bypassed structural validation. | Always check finite values, keys, shapes and numerical tolerance. | This protects invalid-input paths; it is not evidence that these failures were exercised on GPU. |
| P2 | YAML evaluation seeds/counts could differ from fixed values used by the evaluator. | Reject unsupported evaluation and diagnostic settings during protocol validation. | Recorded configurations already comply. |

Implementation links: [rollout](../src/flowvla_rl/libero_rollout.py), [protocol](../src/flowvla_rl/protocol.py), [reporting](../src/flowvla_rl/reporting.py), [runner](../src/flowvla_rl/runner.py).

## Core checks

- **SDE and time grid:** ten adjacent transitions and eleven states remain intact. Subsets retain original indices and times. Gaussian densities sum only real action dimensions and executed positions.
- **GRPO objective:** group-centered rewards, no standard-deviation normalization, five-step means divided by five, episode-balanced chunk averaging, and one optimizer step per complete mixed group.
- **Gradient scope:** cached states and old/reference scores are detached; current rescoring has gradients. Selection changes loss terms rather than introducing timestep-specific parameters. VLM/state projection are frozen, allowing the shared-prefix path.
- **Reference and recovery:** construct the reference from initial SFT before restoring current weights, optimizer and RNG. Reject metadata mismatches. Never reconstruct the reference from updated current weights.
- **Randomness:** locally keyed step selection does not advance rollout RNG. Exhaustive enumeration of 252 subsets on a small model verifies the mean loss/gradient identity.
- **Budget and failures:** count complete attempted groups; collapsed groups consume budget without updating; save final checkpoints even for a collapsed last group or zero updates; never convert failure into a zero success-rate measurement.
- **Evaluation:** same author SFT, ten-step ODE, ten tasks with twenty final episodes each, and final checkpoints. The pinned [LeRobot evaluation loop](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/scripts/lerobot_eval.py) has an explicit horizon cap.

## Validation

- Before the review: **95 CPU tests passed**.
- After fixes: **112 CPU tests passed**, including 17 additions for early termination, nonfinite observations, provenance mismatches, matrix membership, configuration boundaries and resumed timing.
- Three-method CPU integration smoke and toy cache regression gate passed.
- Published artifact hashes, historical source hash, success rates and cost tables were rechecked.
- Python compilation and installation-script shell syntax checks passed.
- [Linux GitHub Actions](https://github.com/ChengruiHan/FlowSift/actions/runs/40154856010) passed for the reviewed implementation.

The historical GPU gate and profile apply to the archived implementation. The fixes were not followed by a new GPU experiment. Original reports and source are retained; a future formal experiment must generate fresh caches, a GPU gate and a protocol lock.

## Remaining boundaries

Real GPU optimizer recovery and simulator reconstruction have not been tested by injecting a live training failure. CPU checkpoint tests do not demonstrate bitwise MuJoCo trajectory replay.

If a process is killed after committing a checkpoint but before writing the timing event, that small interval may remain absent from restored loop time. Preserve separate session/failure resource records. Loop time is not the platform bill.

The experiment has one formal training seed. It does not establish that Uniform-5 generally outperforms Late-5 or Full-10. The historical source retains the old behavior; the reviewed main branch contains the fixes. Their source hashes and protocol approvals are not interchangeable.

## Repository presentation

The repository organization draws on [LeRobot](https://github.com/huggingface/lerobot) for installation and usage documentation, and [CleanRL](https://github.com/vwxyzjn/cleanrl) for readable implementations and reproducible experiments. FlowSift adds a clear research question, measured results, public evidence, CPU CI and contribution guidance. Implementation attribution remains in [UPSTREAM.md](../UPSTREAM.md).
