# 核心代码审查 — 2026-10-25

本轮按最终方案 v3 对 SDE、选步、GRPO、参考模型、rollout、预算、checkpoint、评估和报告链路进行了源码审查，并对发现的边界问题添加回归测试。

结论：在本轮检查和现有测试覆盖范围内，未发现选步索引、五步归一化、共享参数更新或固定 SFT reference 的实现错误。发现并修复了以下问题；这不是对所有 CUDA/模拟器路径的全面正确性证明。

## 修复项

| 严重度 | 触发条件与原行为 | 修复 | 对历史实验的解释 |
|---|---|---|---|
| P1 | rollout 提前 `terminated=True` 但未成功时继续执行；`truncated=True` 只跳出 chunk 内循环，外层仍可能采样 | 任一终止/截断/成功均结束整条 episode，保留已执行动作 mask | 常规 horizon 结束本已受 280 步上限约束。旧日志没有每步 done 原因，不能完全排除早停情形，未据此重写历史结果 |
| P2 | 正式矩阵虽完成，但所选 run-id 可以指向矩阵外的 pilot；报告按 run-id 合并，没有交叉比对模型哈希 | 验证所选 run 属于同一锁定矩阵；CSV、manifest、summary、训练 final policy hash 一致才汇总；检查 seed 和 episode 索引 | 历史全部结果通过新校验，成功率与耗时不变 |
| P2 | checkpoint payload 记录保存前时间，恢复后保留的日志却包含该 checkpoint 耗时，导致成本分解不一致 | checkpoint 事件记录保存后累计时间，恢复时用保留日志补齐 | 本次历史运行没有中断恢复；未改变历史成本 |
| P2 | 初始观测中的 NaN/Inf 可能通过容差比较；完全相同 digest 会绕过结构验证 | 总是检查观测有限性、键和形状，并执行容差比较 | 属于异常输入防护；不声称历史 GPU 已覆盖该异常 |
| P2 | YAML 可改写评估种子/回合数等字段，而实际评估仍用固定值 | 协议校验拒绝与已实现配置不一致的字段 | 已运行配置符合原协议 |

相关文件：[rollout](../src/flowvla_rl/libero_rollout.py)、[protocol](../src/flowvla_rl/protocol.py)、[reporting](../src/flowvla_rl/reporting.py)、[runner](../src/flowvla_rl/runner.py)。

## 核心检查结果

- **SDE 与时间轴**：保留十个相邻转移及十一个状态；子集使用原索引和时间，不跨步拼接。高斯密度按真实动作维度及已执行位置求和。
- **GRPO 目标**：组内中心化奖励、不做标准差归一化；五步损失除以五；按 episode 均衡、chunk 平均。一次完整组只调用一次 optimizer step。
- **参数与梯度**：缓存轨迹和 old/reference detach；current 重评分有梯度；选步改变损失项而非创建不同 timestep 参数。VLM/state_proj 冻结，共享 prefix 的前提满足。
- **reference 与恢复**：先从初始 SFT 构建 reference，再恢复 current/optimizer/RNG；metadata 不一致拒绝恢复。reference 不从已更新 current 重建。
- **随机性**：选步使用独立局部 RNG，不推进 rollout RNG；Uniform 子集的 loss/gradient 均值通过全部 252 子集的 CPU 枚举检验。
- **预算与失败**：以完整 attempted groups 计数；collapsed 组消耗预算而不更新；末组 collapsed 和零更新仍保存 final；失败不能填成 0 成功率。
- **评估与来源**：同一作者 SFT、十步 ODE、十任务每任务二十条、final checkpoint。核对了锁定 LeRobot 版本的 [官方评估循环](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/scripts/lerobot_eval.py)，其循环有显式 horizon 上限。

## 本轮验证

- 修改前：95 项 CPU 测试通过。
- 修改后：112 项 CPU 测试通过，新增 17 项覆盖早停、非有限观测、评估来源错配、锁定矩阵成员资格、配置边界和恢复计时。
- 三方法 CPU 集成 smoke 和 toy 缓存正确性验证。
- 重新核对公开历史产物的文件哈希、训练源码快照哈希、成功率及成本表。
- Python 编译检查和安装脚本 shell 语法检查。

历史 GPU gate 和 profile 仅支持当时的源码。当前分支的修复没有重新开展 GPU 实验；保留原始报告和源码快照，不将旧 gate 用作新代码的验证结果。未来训练必须生成新缓存、GPU gate 和协议锁。

## 剩余边界

- 真实 GPU optimizer 恢复、模拟器重建后的继续采样尚未做实际故障注入。CPU 恢复测试不等于 MuJoCo 轨迹逐位重放。
- 若进程在 checkpoint 提交之后、计时事件写入之前被强杀，该小段 checkpoint 时间仍可能不在恢复后的训练循环累计值中；session/failure 资源记录应单独保留。循环耗时不是平台账单。
- 只有一个正式训练 seed，不能据此宣称 Uniform-5 普遍优于 Late-5 或 Full-10。
- 公开快照中的原始实现保留当时行为，当前主分支才包含修复。二者不得混用来源哈希或协议锁。

## 仓库整理依据

参考 [LeRobot](https://github.com/huggingface/lerobot) 的安装/使用文档组织，以及 [CleanRL](https://github.com/vwxyzjn/cleanrl) 的算法可读性和可复现实验说明：补全快速开始、结果证据、方法说明、CPU CI 与贡献约定。未将这些项目的研究结论移植为 FlowSift 的结论；实现来源仍按 [UPSTREAM.md](../UPSTREAM.md) 记录。
