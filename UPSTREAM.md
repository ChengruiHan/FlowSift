# 来源与修改边界

本项目依据 `FlowSift-final-plan.md`（最终方案 v3.0）实现。

- 底座：[BlackMirean/FlowVLA-RL](https://github.com/BlackMirean/FlowVLA-RL/tree/50e73afc07212f48bed370ea4c88a3131d6f914a)，commit `50e73afc07212f48bed370ea4c88a3131d6f914a`。
- 继承上游 MIT LICENSE、SDE 核、GRPO 公式、SmolVLA prefix、LIBERO 场景恢复、toy 示例及回归测试。上游作者和版权声明保留。
- `tests/upstream/smolvla_sde.py`、`tests/upstream/training.py` 为锁定版本的未修改快照，仅供回归测试。
- 修改 `smolvla_sde.py`：在模型前向前选择原始转移索引；完整 rollout 不变。
- 修改 `training.py`：同一索引同步用于 current/reference/old；独立 RNG、每步数值守卫、计时及梯度范围记录。
- 修改 `libero_rollout.py`：显式冻结 state_proj、稳定 chunk ID、采样模式对齐及中途失败资源计数。
- 修改 `reference.py`：deepcopy 时共享 VLM，避免短暂复制整套 VLM 的显存峰值。
- 新增预算循环、原子 checkpoint、协议校验、服务器入口、profile、诊断和报告工具。
- SDE、GRPO 和均匀时间步子采样思想均不作为本项目原创方法。

模型仅使用 `MorpheusTzz/smolvla-grpo-libero-object` 的
`a120aa9b51f8e55db5ad8284e051963c61a2f5ed/sft-100pct-baseline`；权重不随源码发布。

上游的环境记录及历史实测仅作为安装参考，见 `docs/upstream-environment.md`、
`docs/upstream-setup.md`；其中性能数字不属于 FlowSift 实测结果。

## Public repository review (2026-10-25)

FlowSift repository maintenance and modifications are attributed to ChengruiHan; the inherited Teng Zhu / BlackMirean MIT notice is retained in LICENSE. LeRobot and downloaded model assets retain their own upstream terms.

The public repository organization draws on the explicit installation/usage documentation in [LeRobot](https://github.com/huggingface/lerobot) and the readable, reproducible experiment presentation in [CleanRL](https://github.com/vwxyzjn/cleanrl). These are organizational references, not additional copied algorithm implementations.

Post-experiment corrections cover early episode termination, finite initial observations, evaluation provenance, fixed protocol fields and resumed timing. See [the review](docs/code_review.md). The [published evidence](docs/experiments/seed101/README.md) includes the exact historical training source with its original license; it is separate from the reviewed implementation on main.
