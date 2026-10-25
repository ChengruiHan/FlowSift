# Attribution and modification boundaries

FlowSift implements the selective-rescoring study described in the [v3 protocol](FlowSift-final-plan.md).

## Implementation base

The direct base is [BlackMirean/FlowVLA-RL](https://github.com/BlackMirean/FlowVLA-RL/tree/50e73afc07212f48bed370ea4c88a3131d6f914a), commit `50e73afc07212f48bed370ea4c88a3131d6f914a`.

Inherited components include the MIT license, SDE core, GRPO formulas, SmolVLA prefix handling, LIBERO scene restoration, toy demonstrations and regression tests. The original author and copyright notice are retained.

`tests/upstream/smolvla_sde.py` and `tests/upstream/training.py` are unchanged pinned snapshots used as regression references. They and the historical training archive retain their original comments and strings so the source evidence remains intact. Current project documentation is written in English.

## FlowSift work

| Component | Modification |
|---|---|
| `smolvla_sde.py` | Select original transition indices before denoiser forward passes; preserve full rollout. |
| `training.py` | Use identical indices for current/reference/old; independent keyed selection, numerical guards, timing and gradient-scope records. |
| `libero_rollout.py` | Explicitly freeze state projection, assign stable chunk IDs, align sampling mode, count partial-failure resources and handle early episode endings. |
| `reference.py` | Share the frozen VLM during deepcopy to avoid a transient full-VLM copy. |
| Experiment tooling | Attempted-group budgets, atomic checkpoints, protocol validation, GPU entry points, profiling, diagnostics and provenance-checked reporting. |
| Validation and evidence | Upstream regression tests, subset loss/gradient checks, fixed-cache benchmarks, three-method online comparison and public measurement records. |

SDE conversion, GRPO, uniform timestep subsampling, SmolVLA and LIBERO are not claimed as original inventions of this project. FlowSift's contribution is the implementation, controlled validation and empirical cost–performance study.

## Model source

All methods use `MorpheusTzz/smolvla-grpo-libero-object`, revision `a120aa9b51f8e55db5ad8284e051963c61a2f5ed`, subdirectory `sft-100pct-baseline`. Weights are not included in the source distribution. LeRobot and downloaded model assets retain their own upstream terms.

[Upstream environment records](docs/upstream-environment.md) and [setup notes](docs/upstream-setup.md) are installation references. Their hardware, historical findings and performance numbers are **not FlowSift measurements**.

## Review and repository presentation

FlowSift maintenance and modifications are attributed to ChengruiHan; the inherited Teng Zhu / BlackMirean MIT notice remains in [LICENSE](LICENSE).

Post-experiment corrections cover early termination, finite initial observations, evaluation provenance, fixed protocol fields and resumed timing. See the [review](docs/code_review.md). The [public evidence](docs/experiments/seed101/README.md) includes the exact historical training source with its original license, separate from the reviewed main branch.

The presentation draws on [CleanRL](https://github.com/vwxyzjn/cleanrl) for readable experimental implementations and reproducibility, [LeRobot](https://github.com/huggingface/lerobot) for accessible setup and documentation, and FlowVLA-RL for explicit attribution and evidence boundaries. These are organizational references, not additional copied algorithm implementations.
