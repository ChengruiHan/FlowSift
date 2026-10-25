# 服务器训练指南

本机负责代码和 CPU 测试，真实训练在 Linux + NVIDIA CUDA 服务器进行。已记录实验使用 RTX 4090 D 24GB；其他硬件的实际显存是否足够需通过完整 learner 更新确认。

## 1. 上传与环境

上传整个项目，排除 `.venv`、`__pycache__`、`.pytest_cache` 和本机 toy 结果。不要直接复制 macOS 虚拟环境到 Linux。

```bash
rsync -av --exclude=.venv --exclude=__pycache__ --exclude=.pytest_cache --exclude=results \
  /path/to/FlowSift/ USER@SERVER:/path/to/FlowSift/
```

在服务器项目根目录执行：

```bash
bash scripts/setup_server.sh
source .venv/bin/activate
export MUJOCO_GL=egl
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_arch_list())'
```

脚本使用 Python 3.12 和 LeRobot v0.6.0；优先复用已能运行 CUDA 的 PyTorch/torchvision，仅补齐其他依赖。只有没有可用 GPU 栈时，才安装上游参考的 PyTorch 2.11.0 / torchvision 0.26.0 CUDA 12.8 组合。可通过 `PYTHON_BIN` 指定 Python 路径、`TORCH_INDEX_URL` 指定匹配驱动的官方轮子源。Blackwell 卡应确认架构列表含 `sm_120`。服务器的驱动与 EGL 库由机器镜像提供；缺少时按服务器发行版安装 NVIDIA 驱动及 EGL/OpenGL 运行库。

实际可运行环境以服务器检查为准。安装冲突应统一解决并记录所有方法共用的环境，不能只为某个方法更换依赖。脚本导出 `results/server_environment.json`、`results/server-requirements.lock.txt`、`results/lerobot.commit`。它不会下载示教数据集或启动训练。

正式实验要求源码已经提交 Git，以记录 FlowSift commit；上传的是无 Git 目录时，在服务器建立版本库并提交源码后，再进行 Stage A。不要提交权重和缓存。

## 2. 下载锁定的 SFT

```bash
python scripts/prepare_model.py --output models/smolvla-source
export FLOWSIFT_SFT="$PWD/models/smolvla-source/sft-100pct-baseline"
```

下载固定 HF revision 下的 SFT 子目录，同时对权重、配置、预处理和后处理文件计算 SHA256。训练入口拒绝不匹配的来源或文件。不得使用同仓库已经 GRPO 训练过的 checkpoint 当起点。

## 3. 先验证 SFT ODE 和环境重放

```bash
python scripts/evaluate_libero.py --checkpoint "$FLOWSIFT_SFT" --run-id sft \
  --split replay --output results/replay_a
python scripts/evaluate_libero.py --checkpoint "$FLOWSIFT_SFT" --run-id sft \
  --split replay --output results/replay_b
```

每次覆盖 10 个任务、每任务一个固定条件。检查 `eval_episodes.csv` 中初始观测摘要、成功和步数是否可重复；没有证明严格场景配对时，后续不做 paired bootstrap。ODE 初始高斯噪声仍由固定 seed 控制。

## 4. Stage A：采集独立的 Full 缓存

```bash
python scripts/train_grpo_libero.py --config configs/full10.yaml \
  --stage smoke --seed 17 --budget 20 --sft "$FLOWSIFT_SFT" \
  --run-dir results/stage_a --export-profile-groups results/profile_caches
```

此 run 不计入 pilot／正式预算。第一个有效更新初始化 Adam 状态后，程序从三个不同任务各导出一个 mixed-group 更新前快照，包含完整轨迹、current、optimizer、policy hash 和 reference 来源信息。20 组是采集起始预算，不保证得到三个 mixed 缓存；不足时检查奖励与 mixed 比例，再用新的 Stage-A run 补充。不得把无混合组的缓存冒充有效 profile。

```bash
python scripts/make_diagnostic_cache.py --caches results/profile_caches \
  --output results/fixed_diagnostic_chunks.pt
python scripts/verify_cache.py --sft "$FLOWSIFT_SFT" \
  --cache results/profile_caches/CACHE_FILE.pt --output results/gpu_correctness.json
```

将 `CACHE_FILE.pt` 替换成实际缓存文件名。GPU gate 检查全部十步 old/current、子集切片、原始 Full 重评分、原始 Full 单次更新和梯度参数集合。`atol/rtol` 默认 `1e-6/1e-5`，BF16 容差若需调整，必须先依据原基线重复性测量统一锁定；不得按方法单独放宽。old/current 的 0.01 门槛不能通过该参数改动。

`gpu_correctness.json` 会绑定源码 hash 和 reference hash。之后修改 Python 源码必须重新验证。

## 5. Stage B：同缓存 profile

```bash
python scripts/profile_learner.py --caches results/profile_caches --sft "$FLOWSIFT_SFT" \
  --output results/gpu-profile --gradient-probe
```

默认三缓存 × 三方法 × 三重复，共 27 个完整更新。每轮方法顺序轮换，每个方法使用独立子进程，预热后恢复快照；总时间起止同步 CUDA。16 个固定 chunk 的细分计时和梯度诊断是额外 pass，不混入自然吞吐测量。GPU allocated/reserved 峰值单列，CPU缓存搬运包含在 learner 总时间内。缓存加载及重复间状态恢复不计入更新耗时。

## 6. Stage C：三个 pilot

建议在 `tmux` 会话运行，统一保持环境变量及虚拟环境。单 GPU 按顺序训练：

```bash
python scripts/run_matrix.py --stage pilot --sft "$FLOWSIFT_SFT" \
  --diagnostic-cache results/fixed_diagnostic_chunks.pt \
  --correctness-report results/gpu_correctness.json --run-root results/runs
```

每个方法 seed=17、100 attempted groups、400 episodes。三组独立从 SFT 开始。可先加 `--dry-run` 查看命令。某个 run 失败后不会自动无限重试；排查后重新执行同一命令，会跳过已完成 run，并从未完成 run 的 latest 恢复。

随后以 `--split dev` 评估 SFT 与三组 pilot final，所有评估使用同一入口、batch=1、每任务10条。final 模型路径可从各 run 的 `summary.json` 的 `final_checkpoint` 读取，再追加 `/pretrained_model`。

## 7. Stage D：锁定正式矩阵

```bash
python scripts/lock_protocol.py --full-pilot results/runs/pilot_full10_seed17 \
  --seeds 101,202 --output protocol.lock.json
python scripts/run_matrix.py --stage formal --sft "$FLOWSIFT_SFT" \
  --diagnostic-cache results/fixed_diagnostic_chunks.pt \
  --correctness-report results/gpu_correctness.json \
  --protocol-lock protocol.lock.json --run-root results/runs
```

预算由 Full pilot mixed 比例按方案公式在 300/400/500/600 中选择。资源不足可在打开 test 前改用 `--seeds 101`；若必须统一 B=300，用 `--resource-budget-300-reason '具体资源原因'`。锁文件以独占写入创建，不覆盖已有决定。三方法均保留。

单独恢复例子（其余参数必须与原 run 一致）：

```bash
python scripts/train_grpo_libero.py --config configs/full10.yaml \
  --sft "$FLOWSIFT_SFT" --run-dir results/runs/pilot_full10_seed17 \
  --diagnostic-cache results/fixed_diagnostic_chunks.pt \
  --correctness-report results/gpu_correctness.json \
  --resume results/runs/pilot_full10_seed17/checkpoints/latest.json
```

恢复顺序为 SFT reference → current → optimizer → RNG。环境重建后继续采样；没有声称完整 MuJoCo 轨迹位级恢复。checkpoint 后未提交的日志会保存在 `replayed-work-*.jsonl`，失败和 session 墙钟保留。最新 checkpoint 在完整组边界保存，每50 attempted及 final；中断可能重复该边界之后的采样。

## 8. Stage E：最终评估和汇总

完整正式矩阵全部完成后，评估 SFT 一次以及每个正式 final。每个 checkpoint 共200条、10任务等权、最多280步。

```bash
python scripts/evaluate_libero.py --checkpoint "$FLOWSIFT_SFT" --run-id sft \
  --split test --protocol-lock protocol.lock.json --matrix-runs results/runs \
  --output results/evaluations/sft
python scripts/evaluate_libero.py --checkpoint PATH_TO_FINAL/pretrained_model \
  --run-id formal_full10_seed101 --split test --protocol-lock protocol.lock.json \
  --matrix-runs results/runs --output results/evaluations/formal_full10_seed101
# 对其余正式 final 重复相同调用；run-id 必须与训练目录名称一致。
python scripts/summarize_results.py --runs results/runs --evaluations results/evaluations \
  --profile results/gpu-profile/learner_profile.csv --output results/report
```

保留每个 seed 的主表、逐任务 SR、paired cache 耗时、时间分解和失败记录。汇总不会把 CPU profile 画进真实性能—成本图。进程时间不等于云服务器分配计费时长；空闲、环境安装、故障排查等成本须另行登记。
