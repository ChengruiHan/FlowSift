# 方法与损失

FlowSift 减少的是 learner 重评分的转移数。rollout 仍产生十次 SDE 转移、缓存十一个状态；评估仍使用十步 ODE。`k=0` 对应高噪声端，Late-5 使用 `k=5,…,9`。

## 损失如何计算

同一初态采样 G=4 条 episode，成功奖励为 0 或 1。episode i 的优势：

$$A_i=R_i-\frac{1}{G}\sum_{j=1}^{G}R_j.$$

不除以标准差。对 episode i 的 chunk c，从十个原始转移中选出集合 S（Full 为全部十步，其余为五步）。对每个 k，计算高斯转移的 log probability，沿**已执行的动作位置和真实动作维度**求和；padding 和未执行动作不计入密度。

$$r_{ick}=\exp(\operatorname{clamp}(\ell^\theta_{ick}-\ell^{old}_{ick},-20,20)),\quad d_{ick}=\ell^{ref}_{ick}-\ell^\theta_{ick}.$$

$$L_{ick}=-\min\left(r_{ick}A_i,\operatorname{clip}(r_{ick},0.8,1.2)A_i\right)+0.01\left(e^{d_{ick}}-d_{ick}-1\right).$$

设 episode i 有 C_i 个 chunk，则完整组损失是：

$$L=\frac{1}{G}\sum_i\frac{1}{C_i}\sum_c\frac{1}{|S_{ic}|}\sum_{k\in S_{ic}}L_{ick}.$$

代码逐 chunk backward 累积这个加权损失，全部 chunk 完成后裁剪梯度范数至 1，再调用一次 AdamW。每组只做一个 epoch。old/reference 和缓存轨迹均 detach；梯度流向 current 的共享可训练参数。k3 是在已采样转移上的正则估计，不是整个环境策略 KL 的精确计算。

对应代码：[grpo.py](../src/flowvla_rl/grpo.py)、[training.py](../src/flowvla_rl/training.py)、[smolvla_sde.py](../src/flowvla_rl/smolvla_sde.py)。

## “只更新选中的五步”是什么意思

只为五个转移建立重评分损失和反向计算图，但**不存在五套独立的 timestep 参数**。denoiser 在所有时间共享参数，更新这些参数后，未选中的时间步输出也可能改变。完整 rollout 采用 no-grad，不通过十步生成链反传。

训练仅一个 epoch，且更新前 current≈old，因此初始 ratio≈1，PPO clipping 通常不激活。但 ratio 对 current 参数的导数并不为零，不能因此认为 PG 没有更新。

## 为什么 Uniform-5 本次更好

固定缓存、固定参数时，均匀无放回选五步并除以五，其损失和梯度在选步随机性上的期望等于 Full-10 的十步均值。测试枚举全部 252 个五步子集检查了这一性质。Late-5 则系统性地改变时间步权重。

这提供了研究动机，**不证明在线最终成功率必然更高**：梯度裁剪、Adam 更新和后续采样都是非线性的，随机梯度会改变未来轨迹。当前 Uniform-5 与 Late-5 只相差 5 个成功评估回合，且仅一个训练种子。需要多种子重复才能区分稳定收益与随机波动。
