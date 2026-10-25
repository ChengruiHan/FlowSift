# Upstream environment reference

This is an English rendering of the pinned FlowVLA-RL environment notes. It describes **upstream experiments**, not FlowSift's recorded hardware or measurements. FlowSift's actual dependency versions are in its [run manifests](experiments/seed101/runs/formal_full10_seed101/run_manifest.json).

Use this as a compatibility reference, not as a `pip install -r` file. CUDA PyTorch wheels require an appropriate index. Follow the [FlowSift setup guide](server_training.md) for current project commands and the [pinned upstream repository](https://github.com/BlackMirean/FlowVLA-RL/tree/50e73afc07212f48bed370ea4c88a3131d6f914a) for original context.

## Hardware and system reported upstream

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 5090, 32 GB, `sm_120` |
| CPU / memory allocation | 25 vCPU / 90 GB cgroup quota |
| OS | Ubuntu 22.04 container |

Container tools may show host totals rather than the instance allocation. Read `/sys/fs/cgroup/cpu.max` and `memory.max` for the effective quota.

## Dependency reference

| Component | Recorded version / note |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128; architecture support must include the target GPU |
| LeRobot | v0.6.0 release tag, not a moving branch |
| hf-libero | 0.1.4 |
| MuJoCo | 3.8.1 |
| robosuite | 1.4.0; restoring simulator state does not restore model-level fixture placement |
| imageio / imageio-ffmpeg | Used for upstream rollout video rendering |

## Upstream model and data pins

| Asset | Revision / location |
|---|---|
| `lerobot/smolvla_base` | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| `lerobot/libero` dataset | `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| Upstream 10% / 3% episode manifests | `data/manifests/libero_{10,3}pct.json` in the upstream project |

FlowSift starts from an already trained author SFT and does not need to reproduce upstream dataset subsets.

## Environment variables

```bash
export MUJOCO_GL=egl                    # Headless rendering
export HF_HUB_DISABLE_XET=1             # Optional workaround for Xet problems
# Optional mirror for networks that require it:
# export HF_ENDPOINT=https://hf-mirror.com
# Upstream scripts may use FLOWVLA_RUNS to select persistent storage.
```

Set required variables inside the actual launch script. Detached, noninteractive processes may not inherit interactive shell startup configuration.
