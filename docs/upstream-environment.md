# 钉死的版本清单

跑出本仓库全部结果的环境。**不要用 `pip install -r` 直接装**——
torch 需要 cu128 的专用索引，请按 [README](https://github.com/BlackMirean/FlowVLA-RL/tree/50e73afc07212f48bed370ea4c88a3131d6f914a) 的步骤走；
本文件用于**核对版本是否一致**。

## 硬件与系统
| | |
|---|---|
| GPU | NVIDIA RTX 5090（32 GB，compute capability `sm_120`） |
| CPU / 内存 | 25 vCPU / 90 GB（cgroup 限额，非宿主机数字） |
| OS | Ubuntu 22.04（AutoDL 容器） |

> 容器内 `nproc`(208) / `free`(754GB) / `top` 的 idle% **全是宿主机的数字**。
> 真实上限只能读 `/sys/fs/cgroup/cpu.max` 与 `memory.max`。

## 核心依赖
| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.12.3 | |
| torch / torchvision | 2.11.0+cu128 | `arch_list` 必须含 `sm_120`，否则 5090 跑不了 |
| lerobot | **v0.6.0**（release tag，非 HEAD） | HEAD 的 API 与本仓库不兼容 |
| hf-libero | 0.1.4 | |
| mujoco | 3.8.1 | |
| robosuite | 1.4.0 | `set_init_state` 只恢复 qpos，恢复不了模型级 fixture 位置（见 lab.md E14） |
| imageio + imageio-ffmpeg | 任意近期版本 | 仅 `scripts/render_rollout.py` 需要 |

## 钉死的模型与数据
| | |
|---|---|
| `lerobot/smolvla_base` | `c83c3163b8ca9b7e67c509fffd9121e66cb96205` |
| `lerobot/libero` 数据集 | revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| 10% / 3% 子集 episode 清单 | `data/manifests/libero_{10,3}pct.json`（含 seed 与选择规则） |

## 必需的环境变量
```bash
export MUJOCO_GL=egl              # 无头渲染；缺了会在 reset 时崩
export HF_ENDPOINT=https://hf-mirror.com   # 仅国内网络需要
export HF_HUB_DISABLE_XET=1       # 绕开 Xet 后端故障
export FLOWVLA_RUNS=<持久化目录>   # 否则 RL 产物落到会被回收的盘
```
> 这些变量**必须显式写进启动脚本内部**。`.bashrc` 对 `setsid nohup` 脱离终端的
> 后台脚本无效——这个坑踩过两次。
