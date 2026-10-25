# 单种子实验结果与限制

公开结果、逐回合数据及历史源码快照见 [seed101 证据目录](experiments/seed101/README.md)。下文 `results/server-experiment/` 为完整本地备份，不随 Git 发布。2026-10-25 审查后的代码修复与历史结果的关系见 [代码审查](code_review.md)。

实验于北京时间 2026-10-23 03:44:09 正常完成。完成状态写入后约 3.39 秒，服务器守护触发 AutoDL 关机，命令返回 0。用户以无卡模式开机后，报告、逐回合评估、训练日志及来源记录已同步到本地 results/server-experiment/single_seed/。训练与守护进程均已退出，后台跟进已删除。

## 实验设置

- 起点：MorpheusTzz/smolvla-grpo-libero-object/sft-100pct-baseline，revision a120aa9b51f8e55db5ad8284e051963c61a2f5ed。所有方法使用同一 SFT。
- 设备：RTX 4090 D 24GB；沿用服务器 PyTorch 2.8.0+cu128。环境记录见 results/server-experiment/server_environment.json。
- 三方法 pilot 各 100 attempted groups，seed=17；正式 seed=101，统一预算 300 attempted groups，每组 4 回合，共 1200 训练回合/方法。
- 最终使用各 run 的 final checkpoint，在 LIBERO-Object 上进行确定性 ODE 评估；每个模型 10 任务 × 20 回合，共 200 回合。
- 完整流程包括重放、Stage A、GPU 正确性门槛、固定缓存 profile、pilot、dev、预算锁定、正式训练、test 和汇总。总流程墙钟约 9 小时 49 分钟，不等于平台完整账单时长。

## 最终结果

| 模型 | 测试成功数 | 成功率 | 相对 SFT | 相对 Full-10 | 有效更新 | 训练循环耗时 |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 120/200 | 60.0% | — | — | — | — |
| Full-10 | 127/200 | 63.5% | +3.5 pp | — | 208 | 105.9 分钟 |
| Late-5 | 133/200 | 66.5% | +6.5 pp | +3.0 pp | 201 | 91.7 分钟 |
| Uniform-5 | 138/200 | 69.0% | +9.0 pp | +5.5 pp | 203 | 88.8 分钟 |

逐回合 CSV 已重新核对：每个模型恰好 200 条 test 记录、每任务 20 条、任务与 episode 索引无重复，成功数与 summary 一致。三方法均正常达到统一预算；记录的失败会话耗时和重放组数均为 0。

训练循环耗时包含 rollout、learner、诊断和 checkpoint，不包含模型初始化、最终评估等开销。Late-5 相对 Full-10 减少 13.4%，Uniform-5 减少 16.2%。不能把 learner 局部加速解释为整次实验或租用费用同比下降。

## 固定缓存性能

真实 GPU profile 完成 3 缓存 × 3 方法 × 3 重复。按各缓存的 learner 中位耗时配对：

- Late-5：耗时减少 46.9%–48.0%，加速 1.88–1.92 倍。
- Uniform-5：耗时减少 36.3%–50.8%，加速 1.57–2.03 倍。
- 实际 current/reference denoiser 调用数从每 chunk 10/10 降至 5/5。

profile 的输入快照相同；在线训练则产生不同轨迹和有效更新次数，两类计时分开解释。详细阶段耗时和进程内显存指标保存在 gpu_profile/，不能将其当作整张卡峰值占用。

## 正确性证据与边界

真实 GPU 缓存回归报告为 passed；初始 SFT 的两次小样本 ODE 重放 10/10 条件一致。CPU 95 项测试和服务器测试记录是额外工程证据，不据此声明所有 GPU 故障恢复情形均已实测。

本次结果支持：在该 SFT、seed=101、统一 300 组预算下，两种五步方法均降低训练成本，并取得高于 Full-10 的观察成功率。Uniform-5 本次观察值最高。

限制：

- 只有一个正式训练 seed；200 个评估回合不能代替多个独立训练重复，不能宣称统计显著、普遍优越或统计非劣。
- 未验证官方 SFT 起点或其他任务套件；不能直接与其他评估设置的论文数字比较。
- train/dev/test 沿用 LIBERO 初态分布，未构造严格不重叠的初态划分；不作严格未见场景泛化结论。
- 重放抽查一致不等于所有条件严格配对，当前不报告 paired bootstrap。
- 本次没有实际中断恢复；GPU optimizer 恢复与模拟器重建的异常路径不能宣称已在线验证。
- 2026-10-23 已完整备份模型权重、optimizer 状态、验证缓存、初始 SFT、实际训练代码 bundle、环境记录及日志到本地 results/server-experiment/。共 206 个文件、20,689,756,462 字节，逐文件 SHA256 全部一致；校验完成后再次执行服务器关机，返回码为 0。

## 产物

以下路径均相对 results/server-experiment/single_seed/：

- 主表：report/online_per_seed.csv
- 成本表：report/system_cost.csv
- 固定缓存配对：report/profile_paired.csv
- 逐任务表：report/task_success.csv
- 原始评估：evaluations/*/eval_episodes.csv
- 协议锁：protocol.lock.json
- 完成和关机记录：progress.json、shutdown_guard.json
- 图表：report/task_differences.png、report/success_vs_learner.png、report/training_time.png

## 完整备份与再次关机

2026-10-23 13:10:31（北京时间）完成本地备份校验，并立即执行服务器 sync 与 shutdown，命令返回 0。文件清单见 results/server-experiment/backup_manifest.json，校验结果见 backup_verification.json，关机执行状态见 backup_status.json。源码历史已保存为 server-source.bundle；初始 SFT 位于 models/smolvla-source/；各实验的完整模型和优化器状态位于 single_seed/runs/ 与 single_seed/stage_a/ 下的 checkpoints/。
