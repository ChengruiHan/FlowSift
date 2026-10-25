# FlowSift

[![CPU checks](https://github.com/ChengruiHan/FlowSift/actions/workflows/tests.yml/badge.svg)](https://github.com/ChengruiHan/FlowSift/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Selective denoising-transition rescoring for SmolVLA GRPO on LIBERO-Object.**

FlowSift studies the cost of the learner's probability recomputation. Every method still samples a full **10-step SDE trajectory** during training and uses **10-step ODE evaluation**. The five-step variants rescore fewer stored transitions for the loss; the denoiser parameters remain shared across all timesteps.

Built on [FlowVLA-RL](https://github.com/BlackMirean/FlowVLA-RL/tree/50e73afc07212f48bed370ea4c88a3131d6f914a), using [LeRobot](https://github.com/huggingface/lerobot). See [upstream attribution](UPSTREAM.md), [method and loss / 方法与损失](docs/method.md), and the [v3 research protocol](FlowSift-final-plan.md).

| Method | Rescored transition indices | Current / reference denoiser calls per chunk |
|---|---|---:|
| Full-10 | 0–9 | 10 / 10 |
| Late-5 | 5–9 | 5 / 5 |
| Uniform-5 | Five uniformly sampled indices, without replacement, per chunk | 5 / 5 |

## Single-seed experiment

Author-provided SFT initialization; LIBERO-Object; training seed **101**; **300 attempted groups × 4 episodes** per method; one RTX 4090 D (24 GB). Final ODE evaluation uses 10 tasks × 20 episodes per model.

| Model | Successes / 200 | Success rate | Learner updates | Training loop |
|---|---:|---:|---:|---:|
| SFT | 120 | 60.0% | — | — |
| Full-10 | 127 | 63.5% | 208 | 105.9 min |
| Late-5 | 133 | 66.5% | 201 | 91.7 min |
| Uniform-5 | 138 | 69.0% | 203 | 88.8 min |

Training-loop time includes rollout, learner, diagnostics and checkpointing. It excludes initialization and final evaluation. Relative to Full-10, the observed loop-time reductions were **13.4%** (Late-5) and **16.2%** (Uniform-5). Fixed-cache GPU learner reductions were **46.9–48.0%** and **36.3–50.8%**, respectively.

These are **single-training-seed observations**, not statistical superiority or non-inferiority claims. The evaluation initial-state distribution is not a strict held-out split. Uniform-5's higher observed success rate does not establish why it performed better. Inference speed is unchanged by the selection mechanism.

[Results and limitations / 结果与限制](docs/results_and_limitations.md) · [Public evidence and provenance](docs/experiments/seed101/README.md) · [Core code review](docs/code_review.md)

The recorded GPU experiment predates the code-review fixes on this branch. Its exact source snapshot and source hash are included with the evidence. The reviewed code has CPU regression coverage; it has **not** been rerun on GPU. Changed source requires a new correctness gate and protocol lock.

## Quick start: CPU validation

Python 3.12 is the tested version. No model download, simulator or GPU is needed for these checks.

```bash
git clone https://github.com/ChengruiHan/FlowSift.git
cd FlowSift
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/smoke_cpu.py --output results/cpu-smoke
python scripts/verify_cache.py --toy --output results/cpu-correctness.json
python scripts/verify_published_results.py
```

Use a fresh output directory for each smoke run. Toy timings verify execution and call counts; they are not real-model speed measurements. Linux CI installs the PyTorch CPU wheel explicitly.

To regenerate the published tables and plots:

```bash
python scripts/summarize_results.py \
  --runs docs/experiments/seed101/runs \
  --evaluations docs/experiments/seed101/evaluations \
  --profile docs/experiments/seed101/gpu_profile/learner_profile.csv \
  --output results/reproduced-report
```

## GPU training

Follow the [server training guide](docs/server_training.md) for Linux, CUDA, EGL, LIBERO and the pinned LeRobot stack. The setup script reuses a working installed PyTorch/CUDA stack where available. The recorded experiment used PyTorch 2.8.0+cu128 and LeRobot 0.6.0; complete versions are in the public run manifests.

```bash
bash scripts/setup_server.sh
source .venv/bin/activate
export MUJOCO_GL=egl
python scripts/prepare_model.py --output models/smolvla-source
```

The initialization is `MorpheusTzz/smolvla-grpo-libero-object`, revision `a120aa9b51f8e55db5ad8284e051963c61a2f5ed`, subdirectory `sft-100pct-baseline`. All three methods start from this same author SFT. Weights are downloaded separately and are not included here.

Run order: SFT replay → cache collection → GPU correctness gate → fixed-cache profiling → three pilots → budget lock → formal matrix → final ODE evaluation. The guide supplies the commands. Scripts do not provision paid servers.

## Implementation guarantees

- Original 11 trajectory states are retained. Selected index `k` always scores `states[:, k] → states[:, k+1]` at the original time `1-k/10`.
- Current, old and reference scores use identical selected indices and executed-action masks. Both PG and k3 terms average over the selected count (five or ten).
- Episode-balanced chunk losses accumulate gradients before one optimizer step. VLM and state projection stay frozen; reference weights remain the initial SFT.
- Uniform selection uses a locally keyed RNG and does not advance rollout RNG.
- Collapsed groups consume the attempted-group budget but skip the learner. Complete-group checkpoints retain current weights, optimizer, RNG and protocol metadata.
- Numerical failures stop training. Reports reject mismatched evaluation/checkpoint provenance and retain failed runs without assigning them a zero success rate.

## Repository map

```text
src/flowvla_rl/              SDE, GRPO, selection, rollout, checkpoints, reporting
scripts/                    Training, evaluation, profiling and validation entry points
configs/                    Locked three-method configurations
tests/                     CPU tests and pinned upstream regression fixtures
docs/experiments/seed101/   Evaluation CSVs, logs, hashes and historical source
docs/                      Method, audit, protocol and reproduction guides
toy/                       Inherited lightweight demonstrations
```

[Contributing](CONTRIBUTING.md) · [MIT license](LICENSE). Upstream copyright and regression fixtures are retained. SDE retrofit, GRPO and uniform timestep subsampling are not claimed as original inventions of this project.
