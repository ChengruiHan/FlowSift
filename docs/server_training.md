# GPU training and reproduction guide

Run real training on Linux with an NVIDIA GPU, CUDA and EGL. The recorded experiment used a 24 GB RTX 4090 D. Validate memory requirements on other hardware with a complete learner update, not an inference-only estimate.

The public evidence describes a historical run. A new run uses fresh output directories, source hashes, GPU correctness gates and a protocol lock. Do not reuse the redacted archived lock as approval for changed code.

## 1. Prepare the environment

Clone the repository on the server, or transfer it without local environments and generated results:

```bash
rsync -av --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache --exclude=results \
  /path/to/FlowSift/ USER@SERVER:/path/to/FlowSift/
```

From the project root:

```bash
bash scripts/setup_server.sh
source .venv/bin/activate
export MUJOCO_GL=egl
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())'
```

The setup uses Python 3.12 and LeRobot v0.6.0. It first attempts to reuse working CUDA-enabled PyTorch/torchvision packages. Only when no usable GPU stack is present does it install the upstream reference combination of PyTorch 2.11.0 / torchvision 0.26.0 with CUDA 12.8. Set `PYTHON_BIN` to select a Python executable and `TORCH_INDEX_URL` to select an appropriate official wheel index. Blackwell devices require suitable architecture support, such as `sm_120` for the referenced hardware.

The server image must provide its NVIDIA driver and EGL/OpenGL libraries. Resolve dependency conflicts consistently for all methods, then record the actual working environment. The script exports:

- `results/server_environment.json`
- `results/server-requirements.lock.txt`
- `results/lerobot.commit`

It does not download demonstration data or launch training. Formal runs require a recorded FlowSift Git commit; commit source before Stage A. Never commit weights or caches.

## 2. Download the pinned SFT

```bash
python scripts/prepare_model.py --output models/smolvla-source
export FLOWSIFT_SFT="$PWD/models/smolvla-source/sft-100pct-baseline"
```

This downloads the SFT subdirectory at the fixed Hugging Face revision and hashes weights, configuration and preprocessing/postprocessing files. Training rejects mismatched source records or files. Do not initialize from an already GRPO-trained checkpoint in the same model repository.

## 3. Check SFT ODE replay

```bash
python scripts/evaluate_libero.py --checkpoint "$FLOWSIFT_SFT" --run-id sft \
  --split replay --output results/replay_a
python scripts/evaluate_libero.py --checkpoint "$FLOWSIFT_SFT" --run-id sft \
  --split replay --output results/replay_b
```

Each check covers ten tasks with one fixed condition per task. Compare initial-observation digests, success and step counts in `eval_episodes.csv`. ODE generation still starts from seeded Gaussian noise. Do not use paired bootstrap unless strict scenario pairing has been established.

## 4. Stage A: collect independent Full caches

```bash
python scripts/train_grpo_libero.py --config configs/full10.yaml \
  --stage smoke --seed 17 --budget 20 --sft "$FLOWSIFT_SFT" \
  --run-dir results/stage_a --export-profile-groups results/profile_caches
```

This run is outside pilot/formal budgets. After the first effective update initializes Adam state, the collector exports pre-update mixed-group snapshots from three different tasks. Snapshots contain full trajectories, current weights, optimizer state, policy hash and reference provenance. Twenty groups is an initial collection budget, not a guarantee of three mixed caches. If insufficient, inspect rewards and mixed-group frequency before collecting a fresh Stage-A run. Collapsed groups are not valid substitutes for mixed-group profiling caches.

```bash
python scripts/make_diagnostic_cache.py --caches results/profile_caches \
  --output results/fixed_diagnostic_chunks.pt
python scripts/verify_cache.py --sft "$FLOWSIFT_SFT" \
  --cache results/profile_caches/CACHE_FILE.pt --output results/gpu_correctness.json
```

Replace `CACHE_FILE.pt` with an actual cache filename. The GPU gate checks ten-step old/current agreement, subset slicing, original Full rescoring, one original Full update and gradient parameter scope. Default `atol/rtol` are `1e-6/1e-5`. Any BF16 tolerance adjustment must be justified from baseline repeatability and applied uniformly. These arguments do not relax the 0.01 old/current health threshold.

The gate binds the source and reference hashes. Changing Python source requires fresh verification.

## 5. Stage B: profile identical cached updates

```bash
python scripts/profile_learner.py --caches results/profile_caches --sft "$FLOWSIFT_SFT" \
  --output results/gpu-profile --gradient-probe
```

The default is three caches × three methods × three repetitions, totaling 27 complete learner updates. Method order rotates. Each method runs in a separate subprocess, warms up, then restores its snapshot. Total update timing synchronizes CUDA at both boundaries.

Phase timings on 16 fixed chunks and gradient diagnostics are additional passes, not natural-throughput measurements. Report allocated/reserved memory separately. CPU-to-GPU transfer is included in learner time; loading caches and restoring state between repetitions are excluded.

## 6. Stage C: run the three pilots

Keep the same environment for all methods, preferably in a persistent terminal session. Run sequentially on a single GPU:

```bash
python scripts/run_matrix.py --stage pilot --sft "$FLOWSIFT_SFT" \
  --diagnostic-cache results/fixed_diagnostic_chunks.pt \
  --correctness-report results/gpu_correctness.json --run-root results/runs
```

Each method independently starts from SFT with seed 17, 100 attempted groups and 400 episodes. Add `--dry-run` to inspect commands first. After diagnosing a failure, invoking the same command skips completed runs and resumes an unfinished run from its latest checkpoint; it does not retry indefinitely by itself.

Evaluate SFT and the three pilot finals using `--split dev`, batch size one and ten episodes per task. Read `final_checkpoint` from a run's `summary.json` and append `/pretrained_model` to obtain its model path.

## 7. Stage D: lock the formal matrix

```bash
python scripts/lock_protocol.py --full-pilot results/runs/pilot_full10_seed17 \
  --seeds 101,202 --output protocol.lock.json
python scripts/run_matrix.py --stage formal --sft "$FLOWSIFT_SFT" \
  --diagnostic-cache results/fixed_diagnostic_chunks.pt \
  --correctness-report results/gpu_correctness.json \
  --protocol-lock protocol.lock.json --run-root results/runs
```

The Full pilot's mixed-group fraction selects a common budget from 300/400/500/600 using the protocol formula. A single-seed study uses `--seeds 101`, chosen before opening formal test results. If a uniform B=300 resource cap is necessary, supply `--resource-budget-300-reason 'describe the resource constraint'`. The lock is created exclusively and does not overwrite an existing decision. Keep all three methods.

To resume a particular run, retain all original arguments:

```bash
python scripts/train_grpo_libero.py --config configs/full10.yaml \
  --sft "$FLOWSIFT_SFT" --run-dir results/runs/pilot_full10_seed17 \
  --diagnostic-cache results/fixed_diagnostic_chunks.pt \
  --correctness-report results/gpu_correctness.json \
  --resume results/runs/pilot_full10_seed17/checkpoints/latest.json
```

Restoration order is SFT reference → current → optimizer → RNG. Sampling continues after environment reconstruction; bitwise replay of complete MuJoCo trajectories is not claimed. Work beyond the restored boundary is retained in `replayed-work-*.jsonl`, with failure/session time recorded separately. Checkpoints are written at complete-group boundaries every 50 attempts and at final; interrupted work beyond that boundary may be repeated.

## 8. Stage E: final evaluation and reporting

Once the complete locked formal matrix finishes, evaluate SFT once and every formal final checkpoint. Each has 200 episodes, ten equally weighted tasks and a 280-step cap:

```bash
python scripts/evaluate_libero.py --checkpoint "$FLOWSIFT_SFT" --run-id sft \
  --split test --protocol-lock protocol.lock.json --matrix-runs results/runs \
  --output results/evaluations/sft
python scripts/evaluate_libero.py --checkpoint PATH_TO_FINAL/pretrained_model \
  --run-id formal_full10_seed101 --split test --protocol-lock protocol.lock.json \
  --matrix-runs results/runs --output results/evaluations/formal_full10_seed101
# Repeat for the other formal finals. Each run-id must match its training directory.
python scripts/summarize_results.py --runs results/runs --evaluations results/evaluations \
  --profile results/gpu-profile/learner_profile.csv --output results/report
```

Keep per-seed and per-task results, paired cache timings, time decomposition and failures. Reporting separates CPU profiles from GPU evidence. Process time is not cloud allocation time: account separately for idle periods, installation and debugging.
