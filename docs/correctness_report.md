# 正确性验证记录

2026-10-25 审查：当前代码通过 112 项 CPU 测试，新增边界修复及证据范围见 [代码审查](code_review.md)。历史实验的公开证据见 [seed101](experiments/seed101/README.md)。

2026-10-23 更新：真实服务器实验已完成，GPU 缓存回归门槛为 passed，实际 profile 和 ODE 重放证据已保存在 results/server-experiment/single_seed/。最终结果见 docs/results_and_limitations.md。下表保留最初 CPU 验证时的证据范围，其“服务器待验证”列不代表最新进度。

验证日期：2026-10-22。平台：macOS ARM64 / Python 3.12.14 / PyTorch 2.14.1 CPU。

本地完成 **95 项测试**（包括保留的上游回归），以及以下独立命令：

```text
python -m pytest -q
python scripts/smoke_cpu.py --output results/cpu-smoke
python scripts/verify_cache.py --toy --output results/cpu-correctness.json
python scripts/profile_learner.py --toy --output results/cpu-profile --gradient-probe
python scripts/train_grpo_libero.py --config configs/uniform5.yaml --validate-only
python -m compileall -q src scripts
bash -n scripts/setup_server.sh
```

CPU 三方法集成各完成3个 attempted groups、2次更新、末组 collapsed，全部生成 final。CPU profile 完成3缓存×3方法×3重复共27次更新，计数确认为 Full 每chunk 10/10、Late/Uniform 5/5。它不代表真实 SmolVLA 的速度或显存结果。

| 门槛 | CPU 证据 | 服务器待验证 |
|---|---|---|
| C1 | 未修改上游 Full 与 None/全索引 logp、梯度、loss、参数更新对齐；冻结 state_proj 不改变实际更新 | 真实 BF16 原基线/修复基线重复性和统一容差 |
| C2 | 任意顺序子集等于完整切片，使用原始时间和执行 mask | 真实缓存子集检查 |
| C3 | 对 denoise_step 实际计数：10/10 或5/5 | GPU profile 调用计数 |
| C4 | 非空、唯一、边界检查；常数目标不因步数减少而缩小 | 复用同一实现 |
| C5 | 枚举全部252子集，含非零 k3 和非单位 ratio，平均 loss/逐参数梯度对齐 | 3个代表chunk、每个8个 Uniform 稀疏梯度探针；不做大模型252次枚举 |
| C6 | 固定密度 mask、完整轨迹 detach、reference 无梯度 | 真实预处理/执行 mask 边界 |
| C7 | 十步 old/current 一致；故意破坏后续chunk也会中止且不更新参数 | 所有转移 abs(delta)<=0.01、无 NaN/Inf |
| C8 | 实际梯度集等于声明集；初始 clipped/unclipped PG 梯度一致 | 真实 SmolVLA 参数名/梯度范围 |
| C9 | 选步/诊断不推进 Python、NumPy、torch RNG，同键复现 | CUDA RNG 和模拟器重建条件 |
| C10 | 全collapsed、末组collapsed、mixed、部分组、已达预算恢复的结束与final | 真实环境异常路径与资源记录 |
| C11 | 两次更新与中间保存/恢复的参数逐位一致；RNG恢复，reference hash不变，拒绝元数据不匹配 | GPU optimizer恢复、环境重建后的继续采样 |

此外覆盖协议参数拒绝、预算公式、三配置合法性、profile 按缓存配对且隔离 backend、评估重复/缺失 episode 检查。

CPU 浮点比较采用 atol=1e-6、rtol=1e-5；checkpoint 连续更新比较采用 atol=rtol=0。
`verify_cache.py` 输出 logp 和参数更新的最大绝对误差、相对L2误差到 JSON。GPU 模式使用相同入口，但必须由服务器实际执行后才能记为通过。

不得把本报告表述为“真实模型 C1–C11 全部验收通过”。服务器 CUDA、模型文件 hash、LIBERO/EGL、ODE 重放和在线训练已在本次单种子实验执行；GPU 中断恢复等未发生的异常路径仍没有本次在线实测证据。
