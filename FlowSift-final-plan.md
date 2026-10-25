# FlowSift：面向 SmolVLA 的生成转移子采样强化学习

> **最终研究与实施方案 v3.0｜2026-10-22**  
> **用途：** 可完成、可复现、有明确结论的简历小项目。  
> **必做方法：** Full-10、Late-5、Uniform-5，三组均需真实训练与评估。  
> **主要问题：** 保持完整动作生成过程时，减少 learner 重评分转移，能省多少训练计算、付出多少任务性能代价？  
> **方案归档说明：** 以下为实验前制定的 v3.0 方案。当前实现、实验结果与审查状态见 [README](README.md)；本文中的预期和待执行项保留为原始计划，不作为已完成声明。  
> **版本含义：** 本文锁定项目范围、方法与结题规则；硬件性能、模型权重 hash 和数值实测仍需实施时填写。

本文作为后续实施依据，取代旧版 `FlowSift.md` 中与本文冲突的范围和验收约定。审阅依据保存在同目录 `FlowSift-review.md`。本文中的新模块、配置和命令接口均为待实现规格，不代表上游已经支持。

## 1. 项目决策：先把一个问题做完整

### 1.1 一句话目标

**在 SmolVLA 的 10 步 SDE 训练采样与 10 步 ODE 评估保持不变的条件下，对比全部 10 步、固定后 5 步和均匀随机 5 步的 GRPO learner 更新，测量正确性、任务成功率和实际计算成本。**

最终必须回答四个问题：

1. 只重评分 5 步，是否确实减少 current/reference 前向调用和 learner 用时？
2. 在相同采样 episode 预算下，两种五步方法与 Full-10 的成功率相差多少？
3. 随机五步与固定后五步的表现是否不同，现有数据能支持多强的解释？
4. 局部 learner 加速是否转化为整次训练的机器时间节省？

这四个问题都有允许的负面答案。项目完成不以“超过 Full-10”或“达到论文创新”为条件。

### 1.2 固定范围

| 项目 | 最终决定 |
|---|---|
| 直接底座 | 固定版本 FlowVLA-RL + 已发布 SmolVLA SFT baseline |
| 任务 | LIBERO-Object 原始 10 个任务；每个训练 run 产生一个多任务策略 |
| 方法 | **Full-10 / Late-5 / Uniform-5 三组必做** |
| 训练采样 | 原始 10 步 SDE；eta=0.4 |
| 评估 | 原始 10 步 ODE；Object 每 episode 最多 280 个控制步 |
| 奖励与优势 | 终局二值成功奖励；G=4；组内中心化，不除标准差 |
| 可变因素 | learner 中参与重评分与组合损失的转移索引 |
| 主预算 | attempted groups；同时报告 episode、env steps 和有效 updates |
| 默认正式重复 | 三种方法 × 2 个训练 seeds；资源允许可事先选择 3 seeds |
| 最小结题规模 | 三种方法 × 1 个共同训练 seed；必须标明单 seed 探索实验 |
| 默认正式预算 | 每个 run 至少 300、至多 600 attempted groups，按第 9 节规则统一决定 |
| 主要产物 | 代码、正确性测试、固定缓存 profile、三组在线结果、研究结论 |

不增加 critic、奖励模型、自适应选步网络、任务 credit assignment、新 VLA 架构、并行训练框架或推理加速。Early-5、3/7 步、Flow-GRPO-Fast 移植均不进入本项目验收。

**预算不足时保留三组，统一减少重复次数；不删 Late-5，也不把不同预算的 run 混成主对照。**

预先指定两个主比较：Uniform-5 vs Full-10、Late-5 vs Full-10；Late-5 vs Uniform-5用于解释随机覆盖与固定窗口的差别。无论哪组表现最好，三组数据都进入主表。

## 2. 从第一性原理推导方法

### 2.1 生成计算与更新计算是两件事

一个机器人动作块由噪声经过 K=10 次向量场计算生成。在线 RL 中，先用旧策略生成动作、执行环境并取得奖励；learner 再对保存的转移计算当前策略概率及梯度。

因此，一次训练同时包含两类模型工作：

- **采样工作：** 需要全部 10 次转移，才能得到最终执行的动作块。
- **优化工作：** 在固定缓存轨迹上重新计算每个转移的损失；这些标量损失按步骤求平均。

若优化目标是十个转移项的平均，估计这个平均不必每次都计算十项。这是 Uniform-5 的出发点。它节省的是更新中的可微计算，而非机器人执行或动作生成计算。

### 2.2 为什么优先采用均匀随机五步

固定轨迹、固定模型参数时，记每一步组合损失为 F_k。Full-10 目标为：

\[
L_{10}=\frac{1}{10}\sum_{k=0}^{9}F_k.
\]

从十个索引中均匀无放回抽取五个构成 S，则每一步入选概率为 1/2，因此：

\[
\mathbb E_S\left[\frac15\sum_{k\in S}F_k\right]
=\frac1{10}\sum_{k=0}^{9}F_k.
\]

同样，在常规可交换条件下，子集梯度的期望等于完整缓存目标梯度。**五步平均的分母必须是 5。** 除以 10 会将梯度规模一并缩小，混淆选步与学习率。

代价是估计方差。令 g_k 为固定缓存上的逐步梯度，令

\[
\bar g=\frac1K\sum_k g_k,\qquad
S_g=\frac1{K-1}\sum_k(g_k-\bar g)(g_k-\bar g)^\top.
\]

无放回均匀抽取 m 步的均值梯度满足：

\[
\mathrm{Cov}(\hat g)=\frac{1-m/K}{m}S_g.
\]

K=10、m=5 时系数为 0.1。若各时间步梯度方向接近，子采样可能代价较小；若差异很大，子采样可能使更新更不稳定。这是需要实验验证的取舍。

该无偏性只针对**固定缓存上的上游组合损失**，不证明真实回报策略梯度无偏，不证明 Adam、全局梯度裁剪之后的参数更新无偏，也不保证最终成功率。

### 2.3 Late-5 检验什么

Late-5 固定使用 k=5,6,7,8,9，目标变为后五项的平均，通常不是 Full-10 的无偏估计。它以放弃全时间轴覆盖换取固定窗口更新，是一个低成本的阶段选择对照。

所有时间步共享同一套参数。只对后五步求梯度并不冻结前五步；更新后的模型仍用于下一轮全部十步采样。因此要检查更新后**所有十步**的漂移。

Late-5 比 Uniform-5 好，只能说明该固定窗口在当前模型、任务和预算下表现更好。缺少 Early-5 等对照时，不推导“晚期普遍比早期重要”。

### 2.4 为什么不预设两倍训练加速

以相同轨迹缓存衡量：

\[
T_{\rm learner}(m)\approx T_{\rm fixed}+m\,T_{\rm selected}.
\]

固定项包括条件 prefix、数据搬运、optimizer 等；可选项主要是 current/ref 重评分及对应 backward。五步只减少可选部分。

整次在线训练还包含环境交互、完整 SDE 采样、评估、存盘和诊断。固定工作负载下，若 learner 占比为 f、learner 自身加速倍数为 s，则理想总加速为：

\[
\mathrm{Speedup}=\frac1{(1-f)+f/s}.
\]

在线策略还会改变 episode 长度及 mixed-group 比例，所以真实总时间必须实测。不能把前向次数减少 50% 写成整次训练时间或显存减少 50%。

## 3. 已有工作与个人贡献边界

| 工作 | 本项目借鉴或区别 |
|---|---|
| FlowVLA-RL | 直接继承 SmolVLA SDE 采样、GRPO、轨迹存储、LIBERO 接口和基础评估 |
| DDPO PyTorch | 已有完整采样后逐样本随机打乱时间步、仅训练部分步骤的实现；是 Uniform-5 的直接方法先例 |
| Flow-GRPO 论文 | Denoising Reduction 缩短训练采样网格；本项目保留 10 步采样网格 |
| Flow-GRPO 普通 SD3 代码 | 有 timestep_fraction；本次核对版本按时间轴前缀训练，不等同于均匀随机子集 |
| Flow-GRPO-Fast / MixGRPO | 窗内 SDE、窗外 ODE，并集中优化窗口；本项目保留全部十步 SDE |
| DPPO | 区分 frozen base actor 与部分步使用的 finetuned actor；本项目不采用这种采样时参数路由 |

依据：[DDPO 时间步重排](https://github.com/kvablack/ddpo-pytorch/blob/1958463f020112c9a7bc85768d296daacc2e1b4b/scripts/train.py#L460)、[Flow-GRPO 论文](https://arxiv.org/html/2505.05470v5)、[普通 SD3 代码](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/scripts/train_sd3.py#L345)、[Fast 实现](https://github.com/yifan123/flow_grpo/blob/879042cf5707f8b90daa98d147d7deac2317c5da/flow_grpo/diffusers_patch/sd3_pipeline_with_logprob_fast.py#L135)、[MixGRPO](https://arxiv.org/abs/2507.21802v7)、[DPPO 参数路由](https://github.com/irom-princeton/dppo/blob/cc7234ad7ff39a8f32de3af903606723a16f0648/model/diffusion/diffusion_vpg.py#L148)。

**可以写为个人工作：** 实现 SmolVLA 固定轨迹选择性重评分；修复实验所需恢复和预算语义；建立梯度/概率回归测试；完成三组成本—性能对照；解释失败或加速边界。

**不作为个人原创贡献：** ODE→SDE、GRPO、均匀时间步子采样思想、SmolVLA 架构、LIBERO 环境，以及上游已实现的精度对齐和场景恢复。

## 4. 版本、模型与环境锁定

### 4.1 已核对的来源

| 对象 | 锁定来源 |
|---|---|
| 直接代码底座 | `BlackMirean/FlowVLA-RL@50e73afc07212f48bed370ea4c88a3131d6f914a` |
| SFT 模型仓库 | `MorpheusTzz/smolvla-grpo-libero-object` |
| SFT 子目录 | **`sft-100pct-baseline/`** |
| 模型仓库 revision | `a120aa9b51f8e55db5ad8284e051963c61a2f5ed` |
| LeRobot | `v0.6.0`；安装时解析并记录实际 commit |
| 数据集下载 | 本项目直接在线 RL，不要求重做 SFT 或下载完整示教数据集 |

模型权重和预处理器一起使用；禁止误用同仓库的 `grpo-seed11-update300/` 作为训练起点。发布 SFT 配置已核对：`num_steps=10`、`chunk_size=50`、`n_action_steps=50`、`max_action_dim=32`、真实动作维度为 7。[模型配置](https://huggingface.co/MorpheusTzz/smolvla-grpo-libero-object/blob/a120aa9b51f8e55db5ad8284e051963c61a2f5ed/sft-100pct-baseline/config.json)

执行前填写以下字段，未填齐不进入正式对照：模型与预处理文件 SHA256、FlowSift patch commit、LeRobot commit、Python/PyTorch/CUDA/driver/MuJoCo/robosuite/hf-libero 版本、GPU 型号和显存、CPU/RAM 实际 quota、磁盘目录及容量。上游版本表作为安装参考，成功运行后导出本机实际环境锁文件。[上游环境记录](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/requirements-pinned.md)

### 4.2 硬件与存储

以单张 RTX 5090 32GB 为目标，使用 Linux、支持该卡的 PyTorch/CUDA 和 EGL headless。实际可用显存必须通过一次完整 learner 更新验证，不能拿上游 SFT 或推理显存代替。

只长期保留 SFT 来源记录、每个正式 run 的 final checkpoint、最新可恢复 checkpoint、少量代表性缓存及日志。临时 checkpoint 用轮换策略，原始结果不覆盖。视频只在最终 checkpoint 上录少量成功/失败案例，不在训练过程中持续录制。

## 5. 严格定义优化目标

### 5.1 时间与随机转移

生成时间从噪声端 tau=1 到动作端 tau=0。K=10、h=0.1，k 为 0-based 索引：

\[
\tau_k=1-kh,\qquad k=0,\ldots,9.
\]

给定完整条件 c、带噪状态 z_k，动作专家输出速度 v_theta。沿用上游：

\[
s_\theta=-\frac{z_k+(1-\tau_k)v_\theta}{\tau_k},\qquad
g_k=\eta\sqrt{\tau_k},
\]
\[
q_\theta(z_{k+1}\mid z_k,c)=\mathcal N(\mu_{\theta,k},\sigma_k^2 I),
\quad
\mu_{\theta,k}=z_k-h(v_\theta-\tfrac12g_k^2s_\theta),
\quad \sigma_k^2=g_k^2h.
\]

重评分时 z_k、z_(k+1) 均来自旧策略缓存并 detach，梯度只经过当前速度预测。保持原时间网格和 eta，不重新生成路径、不将后五步重编号成新的五步积分网格。[上游 SDE 核](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/src/flowvla_rl/sde.py)

连续时间边缘一致性需要相应理论假设；模型误差、RL 参数改变和有限步离散不保证实际 SDE 与 ODE 表现一致。主结论以实际 ODE 评估为准。

### 5.2 掩码与 log-prob

对有效动作元素求和：

\[
\ell_{i,c,k}(\theta)=\sum_{j,d}M_{i,c,j,d}
\log q_\theta(z_{k+1,j,d}\mid z_k,c).
\]

M 同时去除填充动作维度以及本 chunk 未实际执行的位置，包括成功提前终止和 280 步上限导致的截断。old、current、reference 使用同一 M、同一 k、同一缓存状态对。

这是固定完整输入状态下的选定坐标条件 log-density 聚合，不是最终动作块的精确边缘似然。沿用上游 masked surrogate，不在本项目证明它对真实回报严格无偏。

### 5.3 优势与每步组合损失

同一任务、同一初始场景采集 G=4 条 episode，奖励 R_i∈{0,1}：

\[
A_i=R_i-\frac1G\sum_r R_r.
\]

不除组内标准差。同一 episode 的各 chunk 共用 A_i；chunk_discount=1。

为与上游数值实现一致：

\[
\rho=\exp\left(\operatorname{clamp}(\ell_\theta-\ell_{old},-20,20)\right),
\]
\[
P=-\min\left(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\right),
\]
\[
d=\ell_{ref}-\ell_\theta,\qquad D=e^d-d-1,\qquad F=P+\beta D.
\]

epsilon=0.2、beta=0.01。PG 的 log-ratio clamp 保持上游行为；同时记录未截断 delta。k3 的 exp 没有相同保护，出现非有限值即判数值失败，不能只给某一组加 clamp 后继续比较。k3 是缓存样本正则，不宣称为任意采样分布下的精确总体 KL。[上游 GRPO](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/src/flowvla_rl/grpo.py)

### 5.4 三组目标

本次更新包含 N 条 episode；episode i 有 C_i 个 chunk。令 S_(i,c) 为本 chunk 选中的索引：

\[
L=\frac1N\sum_{i=1}^{N}\frac1{C_i}\sum_{c=1}^{C_i}
\frac1{|S_{i,c}|}\sum_{k\in S_{i,c}}F_{i,c,k}.
\]

| 方法 | S | 原始时间 tau | current/ref 重评分调用数/每 chunk |
|---|---|---|---|
| Full-10 | {0,1,2,3,4,5,6,7,8,9} | 1.0,0.9,…,0.1 | 各 10 次 |
| Late-5 | {5,6,7,8,9} | 0.5,0.4,…,0.1 | 各 5 次 |
| Uniform-5 | 每 chunk 独立均匀无放回抽 5 个 | 取对应原始 tau，排序后执行 | 各 5 次 |

PG 和 k3 一起子采样。因此 Late-5 同时改变 PG 与正则覆盖；即使表现变化，也不单独归因于 PG。三组均保留完整 rollout 路径。

### 5.5 单 epoch 的实际语义

沿用 epochs=1；一个 mixed group 的所有 chunk 依次 backward，episode 权重累计完后，才执行一次 gradient clipping 与 optimizer.step。此时本轮重评分前策略与采样策略相同，所有 ratio 应接近 1，PPO clip 在该轮梯度计算中基本不激活。

组内优势均值为零，因此更新前加权 PG 标量可接近零；其梯度一般不为零。第一组中 current=reference，k3 也为零。判断学习是否有效应看梯度、参数更新和任务表现，不以首轮 loss 是否非零为依据。[上游 learner](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/src/flowvla_rl/training.py#L120)

## 6. 最小实现规格

### 6.1 文件与修改边界

| 文件/模块 | 必做工作 |
|---|---|
| `smolvla_sde.py` | 给 recompute_log_probs 增加 step_indices；在 denoise_step 循环前选择索引 |
| 新增 `step_selection.py` | 三种模式、索引校验、稳定且独立的选步随机源 |
| `training.py` | 同步切片 old/ref/current；保留上游权重、loss、梯度累积；增加必要计时 |
| `train_grpo_libero.py` | attempted-budget 正常终止、final 保存、分离 current/reference 恢复 |
| runtime 初始化 | 显式冻结 state_proj，记录参数集合，验证实际梯度 |
| 新增 profile 脚本 | 同缓存、同权重、同 optimizer 初态的三组测量 |
| 新增结果汇总脚本 | 合并日志、逐任务/逐 seed 表、时间分解与图 |
| tests | 索引、重评分、梯度、预算、恢复和 RNG 测试 |

SDE 核、原始采样步骤数、reward、成功判据及上游 GRPO 公式不改。公共修复和选步改动分开提交，便于展示个人贡献与回归证据。

### 6.2 选步接口

```python
# 拟议接口；第一版 learner 固定 batch_size=1
def recompute_log_probs(
    model, prefix_pad_masks, prefix_cache, trajectory,
    *, valid_positions=None, step_indices=None,
):
    # None -> range(K)
    # 返回 [1, m]；按传入索引次序返回
    # state_count 始终为 K+1，h 始终为 1/K
    ...
```

索引必须非空、无重复且在 [0,K) 内。每一步读取 `states[:, k]` 和 `states[:, k+1]`，使用 `1-k/K`。不能先计算十步再切 loss，不能缩短 trajectory 冒充五步采样。

第一版固定逐 chunk 处理，不做 batch 时间步并行优化。未来 batch>1 若用公共一维索引，不应继续声称逐样本独立抽样。

### 6.3 随机源

Uniform 索引仅由稳定键决定，例如 `(selection_seed, attempted_group_id, episode_id, chunk_id)`，对键做稳定 SHA256 编码，派生局部 CPU generator，再执行 `randperm(10)[:5]` 并排序。禁止使用进程间不稳定的 Python `hash()`。

该 generator 不接入 torch 全局、环境 reset 或 rollout 噪声流；同一个 chunk 的 current/ref/old 只用一次生成的索引。保存派生算法版本和根 seed，恢复时用相同稳定 ID。测试应证明额外日志和索引生成不改变 rollout 随机数序列。

根selection seed由`SHA256("FlowSift-v3-selection:" + train_seed)`确定；稳定键不包含方法运行顺序、墙钟或当前已完成updates，以免不同collapse率改变索引定义。正式run全部从同一SFT重新开始，不继承pilot参数或optimizer状态。

### 6.4 参数与精度

冻结 VLM 和 state_proj；训练 action expert 及实际位于 denoise 路径中的 action/time projections，具体参数名列表从运行模型导出。三组的 trainable 名称、实际获得梯度的名称完全一致；reference 全部冻结并锚定初始 SFT。

state_proj 在上游 no-grad prefix 路径本来就不获得梯度，显式冻结用于消除声明歧义。先验证冻结前后 Full-10 的实际参数更新等价；不要开放 prefix 梯度扩大项目范围。

current/reference 共用冻结 prefix。采样与重评分沿用同一 rollout_autocast，CUDA 模型前向使用上游 BF16 路径，Gaussian density 按上游在 FP32 中计算。检查 train/eval 模式是否引入 dropout 差异；以 old/current 一致性测试作为最终依据。

### 6.5 一次更新的执行顺序

```text
完整 SDE rollout -> CPU 存储完整 states、element logp、条件与执行 mask
组内计算优势 -> 若全成功/全失败，记录预算后跳过整个 learner
若为 mixed group：
  optimizer.zero_grad()
  对每个缓存 chunk：
    取得三种模式对应的 steps
    在与 rollout 相同精度下构建冻结 prefix
    no_grad：reference 只重评分 steps
    grad：current 只重评分 steps
    old masked logp 取相同 steps
    上游 grpo_loss -> 乘 episode-balanced chunk weight -> backward
  记录 pre-clip grad norm -> 全局裁剪 -> optimizer.step()
  到规定诊断间隔时，对固定小缓存全部十步做 no_grad 漂移检查
```

## 7. 必须修复的预算与恢复行为

### 7.1 预算语义

一组完整采集 G=4 条 episode 后，attempted 增加 1，不论是否 mixed。全成功/全失败组均计入采样成本，并跳过 PG 和 KL 更新，沿用上游组过滤行为。

正式循环以 `attempted < B` 为条件。正常达到 B 后，独立于是否刚发生有效更新，保存 final checkpoint 和 summary。最后一组 collapsed、整个 run 无更新、恢复时已经到 B，都必须正常结束并记录实际状态。

固定 `groups_per_update=1`，不增加 pending-groups 恢复复杂度。正式实验关闭因 collapsed 比例而自动提前终止的启发式规则，改为记录警报；数值、资源或环境异常仍中止。否则不同方法会获得不同预算。

现有上游 max-attempts 是异常上限，不能直接当正常预算实现。[上游循环](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/scripts/train_grpo_libero.py#L145)

环境异常发生在组中途时，该组不作为完成的 attempted group，不对不完整组算优势；已消耗的 episode/环境步/机器时间计入额外资源日志。恢复从最近完整组边界开始，重复采样成本不能隐藏。

### 7.2 current 与 reference 分离恢复

必须使用两个独立来源：

- current：新 run 从 SFT 加载；resume 从已保存的当前策略加载。
- reference：始终从固定 SFT 来源构造并校验 hash；不从 resume current 重新复制。

上游原 resume 分支没有自动恢复 current 权重，直接复用会造成错误。构造顺序应是：确认元数据与 SFT reference → 加载 current → 建立相同参数集合的 optimizer → 加载 optimizer 状态 → 完成初始化后恢复全部 RNG → 从保存的 attempted 边界继续。[上游恢复入口](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/scripts/train_grpo_libero.py#L112)

checkpoint 至少包含 current 权重、optimizer、attempted/episodes/env_steps/updates、训练配置 hash、reference 来源/hash、Python/NumPy/torch CPU/CUDA RNG、选步根 seed 与稳定 ID 方案、累计 collapsed 数和运行时间。每 50 attempted groups 以及 run 结束保存；采用临时目录完整写入后再更新 latest 指针，避免恢复半写文件。

固定缓存 learner 的恢复要求与连续运行数值等价。完整仿真恢复另需验证环境随机源和隐状态；未证明位级重放时，报告“learner 状态精确恢复、环境重建后继续采样”，不能宣称整条 RL 轨迹完全相同。

## 8. 正确性门槛：只做会影响结论的测试

| ID | 必做测试 | 验收依据 |
|---|---|---|
| C1 | Full None / 全索引 / 原实现回归 | 同缓存 logp、loss、逐参数梯度及一次更新对齐 |
| C2 | 子集重评分 | subset 结果等于 full 对应切片，保持原始时间和 mask |
| C3 | 真实调用次数 | 每 chunk current/ref：Full 各10；Late/Uniform 各5 |
| C4 | 随机抽样与归一化 | 合法无放回；枚举252子集的均值与 Full 对齐；常数 loss 不随 m 缩小 |
| C5 | 小模型梯度恒等式 | 枚举子集的平均梯度等于 Full 梯度，不只测标量均值 |
| C6 | mask / detach | 固定 mean/std 下被 mask 的元素不影响聚合；缓存与 reference 无梯度 |
| C7 | old/current 数值一致 | 更新前所有已重评分转移有限且 delta-logp 近0；测试批覆盖全部十步 |
| C8 | 参数范围与单 epoch | 实际梯度集合符合声明；初始 clipped/unclipped PG 梯度一致 |
| C9 | 独立随机源 | 选步、诊断、日志不推进 rollout RNG；重启索引可重现 |
| C10 | 预算边界 | collapsed/全collapsed/末组collapsed/已到预算恢复均正确保存与结束 |
| C11 | 恢复与 reference | 固定缓存连续两次更新 vs 中间保存恢复一致；reference hash 不变 |

CPU 小模型起始容差：`atol=1e-6, rtol=1e-5`；梯度和参数比较同时报告最大绝对误差及相对 L2 误差。真实 BF16 测试先跑原始基线测量数值重复性，再在选步实现前记录统一容差；不能为某一种方法单独放宽。

GPU old/current 健康门槛建议固定为所有测试转移 `abs(delta_logp) <= 0.01` 且无非有限值，意味着 ratio 误差约在 1% 内；目标仍是相同路径尽可能逐位复现。若原始基线不能满足，先定位精度、模式、cache 或 batch 差异，不能带着失配解释算法结果。每次在线更新都记录更新前最大误差，超门槛中止并保存诊断。

mask 测试不要求改变输入 latent 的无效坐标后模型输出不变，因为网络内部可能耦合各坐标。测试对象是密度聚合规则。

共同工程修复可能改变参数列表，但不应改变未恢复 run 的实际更新。先做“原上游 → 共同修复 Full”回归，再做“共同修复 Full → 选步接口 Full”回归，避免把修复收益算作 FlowSift 收益。

正确性门槛失败时，只能形成故障审计记录，不能把尚未成立的算法比较算作完成。

## 9. 实验矩阵、预算与有限停止规则

### 9.1 固定实验顺序

| 阶段 | 方法与 seed | 工作量 | 目的与结束条件 |
|---|---|---|---|
| A：环境与正确性 | 原上游、共同修复 Full、三个选步模式 | 少量 rollout；完成 C1–C11 | 确认确实在优化同一目标的不同步骤集合 |
| B：固定缓存 profile | 三组全部；同一缓存/权重/optimizer | 至少3个 mixed-group 缓存，细节见第11节 | 得到独立于在线策略表现的计算结果 |
| C：三组 pilot | Full/Late/Uniform，seed=17 | **每组100 attempted = 400 episodes** | 检查稳定性、学习信号和实际成本 |
| D：正式主对照 | 三组全部，seeds=[101,202] 默认 | 每个 run 相同 B，B∈{300,400,500,600} | 默认6个独立训练run；固定 final checkpoint |
| E：最终评估与结题 | SFT及所有正式 final checkpoints | 每 checkpoint 200 ODE episodes | 三组性能表、profile、系统时间表和结论 |

Stage A 中生成 profile 缓存可消耗额外 Full rollout，单独计账；不得把这些步骤悄悄并入某一正式方法。

### 9.2 正式 B 的决定规则

在查看正式 test 之前，从 **Full-10 pilot** 计算：

\[
\hat p=\frac{\text{mixed groups}}{100}.
\]

以约 200 次有效更新作为预算规划目标，而非承诺，统一选择：

\[
B=\min\left(600,\max\left(300,
100\left\lceil\frac{200}{100\max(\hat p,0.1)}\right\rceil
\right)\right).
\]

例如 p_hat=0.8 选择300；0.6选择400；0.4选择500；0.2触及600上限。该规则只根据基线的学习信号数量规划成本，不根据哪种五步方法表现好选择预算。

真正有效更新数随策略而变，不会被该公式保证；所有方法仍按同一个 B 结束。记录 `planned_target_updates=200`、实际 updates 和是否受600上限约束。

Full pilot 的 dev 成功率变化用于判断结果解释：若 Full 没有学习信号，正式结果即使三组接近，也要标注“基线收益尚未充分建立”。**不反复延长训练直到得到正结果。**

### 9.3 资源不足时的预定收缩顺序

Stage C 完成后、Stage D 开始前，形成一份不可变 `protocol.lock.json`，选择以下一种规模：

1. **标准结题：** 三方法 × seeds=[101,202]，使用公式选择的 B。
2. **最小结题：** 三方法 × seed=[101]，仍使用相同 B；明确标注单 seed。
3. 如果单 seed 的建议 B 也超预算，统一采用 **B=300、三方法 × seed=[101]**，登记资源原因及有效学习信号可能不足。
4. 如果连三组各300 attempted及最终评估都无法承担，则完成已有三组 pilot 与 profile 后，按“pilot结题”报告，不宣称完成正式训练对照。

可选第三个 seed=303 必须在正式 test 打开前决定，并同时用于三种方法。禁止只补表现差/表现好的方法，禁止评估后挑选种子。

执行正式训练时先完成 seed101 的三组，再完成 seed202 的三组；所有最终 test 在本次锁定的训练矩阵完成后统一运行。这样遇到资源中断时至少保留一套完整三组数据。若临时只能完成较小矩阵，记录协议偏离、原因与未完成run，完整保留已有数据。

### 9.4 什么时候停止

| 情况 | 动作 | 可以得出的结论 |
|---|---|---|
| 正确性失败、old/current失配、NaN/Inf | 暂停该run，定位共同实现；修复后废弃受影响结果并从一致起点重跑 | 暂无有效算法对照 |
| 方法在相同数值路径下失稳，重放同一缓存/恢复点再次复现 | 保留失败并报告稳定性代价；不对它单独调参 | 该固定配置下方法不稳定 |
| Full pilot mixed=0 | 核验奖励/场景/采样一次；实现无误则停止正式扩展 | 本预算/起点下无PG学习信号，选步行为收益无法评估 |
| Full SR未提升，但有非零梯度与有效更新 | 允许按锁定预算完成，结论标注基线收益未建立 | 可报告成本与观察到的性能，不能声称保留RL提升 |
| 五步方法成功率下降 | 正常完成计划；报告性能—成本取舍 | 有效负结果 |
| learner不加速或端到端不加速 | 完成至少三组pilot/评估，资源允许按锁定正式规模完成 | 缩小调用数没有带来所期望的实际收益 |
| 已完成锁定矩阵 | 立即转结题 | 不增加方法分支寻找更好结果 |

硬错误的恢复重试从同一个有效checkpoint进行；失败耗时单独报告。若更改学习率、eta、epoch、KL或数据分布，属于新的协议版本，不能与旧结果拼接成主表。

同一错误最多做一次相同条件重放用于定位；确认可复现后必须修复原因或记录失败，不能反复重启期待偶然通过。每个seed内部三方法执行顺序事先轮换：seed101按Full/Late/Uniform，seed202按Late/Uniform/Full；若使用seed303则按Uniform/Full/Late。

## 10. 训练与评估协议

### 10.1 训练调度

所有方法同一训练seed使用相同 attempted-index 调度：10任务轮换，初态索引按上游规则轮换；G=4共享同一初始场景，episode使用不同policy噪声。记录 task_id、init_state_id、reset_seed、policy_seed。

训练随机数与方法名称解耦，Uniform另设独立selection seed。各方法轨迹会因参数不同逐渐分叉，因此相同seed只能提供共同初始随机条件，不能保证训练数据始终相同。

组内场景一致性沿用并验证上游恢复逻辑；不能仅因为init_state_id相同就认为完整场景相同。发生组内一致性错误时，该组不能进入learner。

### 10.2 默认采用同分布评估，不扩大为泛化研究

第一版不实现新的train/dev/test初态严格分割。沿用LIBERO-Object初态分布，dev和test使用事先固定且不同的评估随机条件；它们可能与RL/SFT见过的场景重叠。

因此结果定义为**同分布任务表现与计算效率**，不声称held-out场景泛化。dev/test的区分指它们在决策过程中的用途，不能仅凭不同seed宣称场景分布独立。

| 项目 | dev | 最终test |
|---|---|---|
| 用途 | pilot健康检查、预算解释；不按方法单独调参 | 一次性评估锁定矩阵 |
| 任务 | Object全10任务 | Object全10任务 |
| 每任务episodes | 10 | 20 |
| 总episodes/checkpoint | 100 | 200 |
| 评估seed | 2000 | 1000 |
| 采样 | 原始10步ODE | 原始10步ODE |
| 步数上限 | 280 | 280 |
| batch size | 1，三组相同 | 1，三组相同 |
| checkpoint | 初始SFT与各pilot final | 初始SFT与各正式run final |

ODE表示积分过程中不额外注入SDE噪声；初始高斯噪声仍需固定。不能把ODE误解为与初始噪声无关的确定动作。

复用上游官方评估调用与预处理，仅增加必要元数据输出；所有方法采用同一版本、batch size、精度与seed。SFT test评估一次供所有方法参考，最终checkpoint不按dev峰值挑选。[上游评测协议](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/configs/eval_protocol.md)

### 10.3 评估可重放性的最低要求

保存完整命令、checkpoint hash、task/episode索引、环境seed、可获得的init_state_id和成功/步数。若能以轻量日志钩子取得初始观测，记录摘要并抽查相同条件的初始图像和本体状态。

至少对SFT的一小组10个评估条件重复执行一次，核验重放误差。若相同名义seed仍产生不同场景或结果，记录这一限制；使用普通逐任务结果汇总，不做需要严格场景对应的paired bootstrap。不能为了得到窄置信区间，默认跨方法episode一一配对成立。

完全保存/恢复MuJoCo隐藏状态不作为本项目新增工程模块；它若成为主要障碍，保留同分布评估与明确限制即可。

## 11. 固定缓存计算实验：优先拿到确定的工程结论

### 11.1 缓存与初态

从Full基线保存至少3个mixed-group缓存，覆盖至少3个任务，且包含成功/失败episode。保存条件输入、完整11个states、old element logp、执行mask、advantage、episode/chunk ID、采样时policy hash。

使用已完成预热、具有Adam状态的基线模型重新采集这批缓存；保存该时刻current/reference/optimizer快照。每个测量重复均从**同一个快照**恢复，避免连续更新后old/current不再一致。

首轮初始化与optimizer状态分配单独测；稳态profile使用已经初始化的状态。CPU缓存到GPU的搬运属于learner工作，单列且纳入learner总时间；从磁盘加载整个实验快照和跨重复恢复不计入单次learner更新。

### 11.2 最小测量规模

- 细粒度：选取16个固定chunk，记录prefix、ref/current、backward耗时与调用数。
- 整体：3个完整group缓存，每种方法每个缓存3次计时，共27个完整learner测量。
- 预热：每种模式至少1次不计时预热，然后恢复快照。
- 顺序：按轮换顺序执行三组，避免始终让某方法受冷启动影响。
- 统计：先报告每个缓存上的重复中位数，再报告跨缓存的配对耗时比及范围；3个重复不用于强统计推断。

总learner用墙钟计时，并在起止执行CUDA同步。细分GPU算子采用CUDA events或单独profile pass；不要在每个小模块强制同步后，又把受干扰的总时间当自然吞吐。

### 11.3 profile指标

| 指标 | 定义 |
|---|---|
| learner ms/update | 搬运+prefix+ref+current+loss/backward+clip+optimizer；排除诊断与checkpoint |
| re-score调用数 | current/ref分别计数；不把rollout或诊断调用混入 |
| learner时间降低 | `1 - T_method / T_full` |
| learner加速倍数 | `T_full / T_method` |
| allocated/reserved峰值 | 独立模式进程测量，预热后重置峰值；另报初始化峰值 |
| 全路径缓存大小 | 三组保持相同，避免将删缓存包装成选步收益 |

每种方法的资源峰值优先用独立进程，避免Full先运行后CUDA reserved缓存污染五步方法。若硬件噪声导致同缓存耗时比跨度超过10%，只允许增加到每缓存5次重复；仍波动大则报告不稳定性，不无限benchmark。

Stage B能够支持“实际减少了重评分计算”，不能独立支持“保持策略成功率”。

## 12. 低成本机制诊断

### 12.1 必做：小模型精确验证，真模型稀疏检查

精确的252子集平均梯度检查只在小模型上做，属于C5。真实SmolVLA不枚举252次大模型梯度。

真实模型选择3个代表chunk，每个chunk固定一次Full梯度、一次Late梯度及8个Uniform子集梯度。只在profile阶段做，单独计时，报告梯度范数和与Full的余弦相似度。若Full梯度近零，余弦指标标记不可解释。

这8个子集仅用于观察方差和方向，不构成真实模型无偏性的数值证明。实现时可用逐参数内积累计，不复制全部梯度矩阵、不保留十步独立全模型梯度集合。

### 12.2 更新后十步漂移

选用同一批8个固定诊断chunk，每个run在attempted=0、每50 attempted以及final进行no-grad全十步重评分。诊断缓存来自正式test之外，并在所有方法间固定。

记录每个k的delta-logp、ratio分位数、k3及非有限值。此处current与缓存old不再相同，ratio不要求接近1；它是漂移指标，不能套用更新前健康门槛。

如容易取得高斯均值，可额外报告固定状态上的解析条件KL：

\[
D_k^{\rm conditional}=\sum_{j,d}M_{j,d}\frac{(\mu_{\theta,k,j,d}-\mu_{ref,k,j,d})^2}{2\sigma_k^2}.
\]

它不替换训练k3，也不是最终动作边缘KL。额外接口若明显增加工作量，首版只做已有logp诊断即可。

晚期更低的转移方差可能放大速度预测误差的概率效应，所以Late-5的“更有用”不能由低噪声直觉推出。三组所有诊断频率一致，诊断耗时独立记录。

## 13. 指标、统计与结论规则

### 13.1 三类核心量

1. **任务性能：** 每个checkpoint的10任务等权ODE成功率、逐任务SR、相对同seed Full的差值（pp）、相对初始SFT的差值。
2. **计算性能：** 固定缓存learner耗时比、调用数、GPU峰值；这是计算优化的主要证据。
3. **系统成本：** 完整训练墙钟、真实GPU分配时长、env steps、episodes、有效updates、collapse率、评估和故障重试成本。

episode数相同不代表环境步数相同。报告中使用“等episode采样预算”，不把结果称为严格env-step样本效率。

### 13.2 稳定性日志

最低记录每组reward、mixed标志、初始delta-logp最大绝对值、raw log-ratio clamp命中率、k3最大d、NaN/Inf、pre-clip梯度范数与是否裁剪。参数update norm按诊断间隔抽测，避免每次复制完整模型影响吞吐。

选步索引要可重建；日志保存稳定ID及派生版本，可仅对少量chunk显式打印索引。出现故障时保存该组完整索引与异常张量摘要。

### 13.3 不把小样本包装成统计证明

- 单seed：报告原始SR与逐任务结果，称“单seed观察”；不称稳定提升。
- 双seed：报告两个种子的所有数值及范围，可报告均值；不依赖n=2的训练seed置信区间。
- 三seed：报告逐seed、均值与标准差，仍说明重复数有限。
- 同一checkpoint的200个episodes不是200次独立训练重复。
- episode级置信区间只描述当前checkpoint在评估条件上的不确定性，不能替代训练seed变异。
- 配对场景经过核验时，才使用场景/任务分层paired bootstrap；否则只作保守描述。
- 不用`p>0.05`、区间跨0或点估计接近证明等价/非劣。

### 13.4 预先定义的工程评价尺度

下列阈值是项目的**描述性工程目标**，不构成统计检验：

| 判断量 | 预设目标/解释 |
|---|---|
| 五步调用数 | current/ref各减少50%，必须由计数确认 |
| learner时间降低 | 中位数降低≥20%记为达到工程目标；0–20%为有限收益；≤0%为未加速 |
| SR点估计差异 | 相对Full下降不超过3 pp记为落在预设容忍带；不能写“统计非劣” |
| 跨seed一致性 | 每个已完成seed分别给出是否满足目标；不能只报均值掩盖一个seed大幅退化 |
| 端到端 | 原样报告百分比变化，不预设必须加速 |

在约200次评估下，3 pp只有6个成功episode之差，因此必须保留小样本解释。即使未达到上述工程目标，只要对照有效，项目仍可结题。

### 13.5 所有主要结果都有对应结论

| 观察 | 合理结论 | 不能推导 |
|---|---|---|
| 五步更快、SR点估计接近且Full有提升 | 本协议下观察到有用的性能—成本权衡 | 普遍无损、统计非劣、两倍总加速 |
| 五步更快、SR明显下降 | 减少更新计算存在可测性能代价 | 方法全面优于基线 |
| Uniform优于Late | 随机覆盖在当前预算更稳妥/点估计更好 | 所有固定窗口都差 |
| Late优于Uniform | 固定末段在当前实验表现更好 | 后期步骤普遍最重要 |
| learner更快、整次训练无收益 | 系统瓶颈限制局部优化收益 | 训练效率已整体提升 |
| Full及五步均未建立SFT以上收益 | 本预算内没有充分观察到RL增益，成本结果仍可报告 | 保留了有效RL提升 |
| 两种五步均不加速 | 当前实现和硬件下调用减少未转化为实际收益 | 理论调用减少等于加速 |
| 某方法发生数值失败 | 该锁定配置下的稳定性问题 | 把失败run当零成功率正常结果平均 |

## 14. 配置草案与日志规格

以下YAML是需要实现的配置合同。主程序必须保存解析后的实际配置，并拒绝恢复时方法、预算语义、模型来源等关键字段不一致的run。

```yaml
project: FlowSift
protocol_version: v3.0
status: planned_not_executed
upstream:
  repo: https://github.com/BlackMirean/FlowVLA-RL
  commit: 50e73afc07212f48bed370ea4c88a3131d6f914a
model:
  repo: MorpheusTzz/smolvla-grpo-libero-object
  revision: a120aa9b51f8e55db5ad8284e051963c61a2f5ed
  subdirectory: sft-100pct-baseline
  reference_source: immutable_initial_sft
  denoising_steps: 10
  chunk_size: 50
  n_action_steps: 50
  freeze_state_projection: true
train:
  suite: libero_object
  task_ids: [0,1,2,3,4,5,6,7,8,9]
  group_size: 4
  groups_per_update: 1
  eta: 0.4
  learning_rate: 1.0e-6
  optimizer: AdamW
  betas: [0.9, 0.999]  # 上游脚本直接调用torch AdamW，非SFT config中的betas
  eps: 1.0e-8
  weight_decay: 1.0e-10
  clip_epsilon: 0.2
  kl_beta: 0.01
  epochs: 1
  max_grad_norm: 1.0
  chunk_discount: 1.0
  advantage_std_normalization: false
  learner_chunk_batch_size: 1
  budget_unit: attempted_groups
  pilot_budget: 100
  formal_budget: TO_LOCK_300_400_500_OR_600
  collapsed_group_early_stop: false
selection:
  methods: [full10, late5, uniform5]
  uniform_count: 5
  sample_per: chunk
  without_replacement: true
  independent_keyed_rng: true
  retain_original_time_grid: true
  normalize_by_selected_count: true
evaluation:
  sampler: ode
  denoising_steps: 10
  max_env_steps: 280
  batch_size: 1
  dev_seed: 2000
  dev_episodes_per_task: 10
  test_seed: 1000
  test_episodes_per_task: 20
  checkpoint: final
  strict_held_out_claim: false
seeds:
  pilot: [17]
  formal: [101, 202]  # 在打开test前按资源规则锁定
  selection_seed_derivation: sha256_v1
diagnostics:
  every_attempted_groups: 50
  fixed_chunks: 8
  score_all_10_steps: true
checkpoint:
  every_attempted_groups: 50
  always_save_final: true
```

AdamW默认betas/eps在环境锁定时再次读取实际optimizer确认；不能误把SFT配置中的优化器设置当在线RL脚本设置。[上游optimizer构造](https://github.com/BlackMirean/FlowVLA-RL/blob/50e73afc07212f48bed370ea4c88a3131d6f914a/scripts/train_grpo_libero.py#L117)

日志分成三个文件，避免每个group重复写全部元数据：

```text
run_manifest.json:
  protocol_hash, code_commits, model_revision, model_hashes,
  dependency_versions, hardware, method, train_seed, budget,
  trainable_parameter_names, reference_hash, selection_rng_version

groups.jsonl:
  attempted_id, task_id, init_state_id, reset_seed, policy_seeds,
  rewards, mixed, episode_count, env_steps, completed_updates,
  chunk_count, pg_loss, k3_loss, max_abs_initial_delta_logp,
  raw_ratio_clamp_fraction, max_k3_d, nonfinite_count,
  grad_norm_pre_clip, gradient_clipped,
  rollout_seconds, learner_seconds, diagnostic_seconds,
  checkpoint_seconds, elapsed_seconds

eval_episodes.csv:
  run_id, checkpoint_hash, split, task_id, episode_index,
  evaluation_seed, init_state_id_if_available,
  initial_observation_digest_if_available, success, env_steps
```

额外结果：`learner_profile.csv`、`step_diagnostics.csv`、`failures.jsonl`、`summary.json`。failed run不得只从汇总目录消失。

## 15. 时间和算力预算

本方案不预承诺GPU小时数或花费。Stage C测得速度后，再结合实际剩余机器时间和费用额度决定正式规模。

定义每个方法的纯采样平均时间 t_rollout_group、mixed比例 p、有效learner平均时间 t_learner_update。训练估算为：

\[
T_{\rm train}\approx B\,t_{\rm rollout\_group}
+B\,p\,t_{\rm learner\_update}
+T_{\rm diag}+T_{\rm checkpoint}.
\]

上游`group_s`可能已经包括learner；必须新增纯采样timer，不能再把group_s与learner相加造成重复计时。

默认双seed、三方法的采样数量如下：

| 阶段 | episodes |
|---|---:|
| 三组pilot，B=100 | 1200 |
| 正式三方法×2seeds | 24B：B=300时7200，B=600时14400 |
| pilot dev：SFT一次+三个pilot final | 400 |
| 最终test：SFT一次+六个正式final | 1400 |
| 环境/缓存/重放检查 | 另计实际数量 |

最小单seed正式对照的训练episodes为12B，最终test为800。上述评估也消耗机器时间，不能只估训练成本。

Stage C后填写：已耗机器小时、剩余额度、三组各自预测时间、评估预测时间、存储用量。剩余预算中建议留20%用于最终评估、结果整理和恢复；这是工程缓冲，不是速度估计。

如果受限的是总预算，先从2seeds缩到1seed，三组都保留；如果受限的是单次run时间，统一B=300。不要把方法数量越加越多，也不要为了更漂亮的数字不断增加epochs。

## 16. 结果产物和展示方式

### 16.1 必须交付的三张表

**在线任务表现：每个seed一行，不先隐藏在均值中。**

| Method | Seed | Attempted | Episodes | Env steps | Updates | ODE SR | Δ vs同seed Full(pp) | Δ vs SFT(pp) | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| SFT | — | 0 | 0 | 0 | 0 | 待测 | — | 0 | 待测 |
| Full-10 | 101 | B | 4B | 待测 | 待测 | 待测 | 0 | 待测 | 待测 |
| Late-5 | 101 | B | 4B | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Uniform-5 | 101 | B | 4B | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| 其余已锁定seed | … | … | … | … | … | … | … | … | … |

**固定缓存计算：**

| Method | Current/ref调用每chunk | Learner ms/update | 时间降低 | 加速倍数 | 稳态allocated峰值 | 初始化峰值 |
|---|---|---:|---:|---:|---:|---:|
| Full-10 | 10 / 10 | 待测 | 0 | 1.00× | 待测 | 待测 |
| Late-5 | 5 / 5 | 待测 | 待测 | 待测 | 待测 | 待测 |
| Uniform-5 | 5 / 5 | 待测 | 待测 | 待测 | 待测 | 待测 |

**端到端机器成本：**

| Method/seed | Rollout | Learner | 诊断 | 保存/其他 | 训练总时长 | 最终评估 | 重试额外成本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 各run | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

时间分解之和应能解释训练总墙钟；有未分类开销时保留“其他”，不能通过选择性计时制造端到端加速。

### 16.2 三张图足够

1. 成功率—固定缓存learner成本散点图：三方法，显示每个seed；明确成本来自固定缓存而非推理。
2. 三方法训练时间分解图：rollout、learner、诊断/保存。
3. 10任务成功率差值图：Late与Uniform分别相对Full。

可附一张十步漂移曲线帮助解释Late行为。视频只辅助展示行为，不能代替统计结果；同时保留至少一个失败例，不只挑最漂亮的成功片段。

### 16.3 仓库结构

```text
FlowSift/
├── README.md
├── UPSTREAM.md
├── LICENSE
├── src/flowvla_rl/          # 保留上游结构，明确标识新增/修改
├── scripts/
│   ├── train_grpo_libero.py
│   ├── profile_learner.py
│   └── summarize_results.py
├── configs/                # 三方法与锁定协议
├── tests/
├── docs/
│   ├── research_protocol.md
│   ├── correctness_report.md
│   └── results_and_limitations.md
└── results/
    ├── manifests/
    ├── tables/
    └── figures/
```

维护README中的实际可执行命令，并在干净环境验证一次最短复现路径。保留上游许可和归属；模型文件使用下载位置/hash管理，避免提交巨大权重。

## 17. 验收、结题与简历表达

### 17.1 完成条件分级

**最低完整项目：** C1–C11通过；三种方法全部实现；三组固定缓存profile完成；至少一个共同正式seed下三组各完成相同B≥300的在线训练与最终ODE评估；提交三张结果表和明确结论。

**标准项目：** 最低完整项目基础上，完成三组×2seeds；结果逐seed可追溯，预算与恢复逻辑经验证，提供实际可复现命令。

**增强项目：** 三组×3seeds；只在预算允许且事先决定时做，不作为必须完成的小项目门槛。

**pilot结题：** 只完成三组100 attempted的pilot与profile时，明确写“工程原型与探索实验”；可以作为简历工程经历，但不写成完成充分RL效果验证。因正确性门槛未通过而停止，则是故障调查产物，不计作有效方法对照。

只要实验有效，无显著提升、策略退化或仅局部加速都不构成项目失败。结题报告必须明确：哪部分已证明、哪部分只是观察、哪部分因预算不能回答。

### 17.2 固定结论模板

```text
在[代码/模型版本]与LIBERO-Object同分布评估协议下，
对Full-10、Late-5、Uniform-5进行了[种子数]个训练seed的比较，
每个run使用[B]个attempted groups=[4B]条完整采样episode。

Full/Late/Uniform的最终ODE成功率分别为[原始逐seed数据]。
相对Full，Late/Uniform的固定缓存learner时间变化为[...]，
完整训练时间变化为[...]；实际有效更新数和env steps为[...]。

本实验支持[成本/性能/稳定性方面的具体结论]。
由于[训练seed数量、同分布场景、基线学习信号等限制]，
不支持[统计非劣、泛化、普遍晚期重要性等更强结论]。
```

### 17.3 简历写法

**所有有效结果都可以使用的基础版：**

> **FlowSift｜SmolVLA 强化学习训练效率研究**  
> 基于 FlowVLA-RL 实现生成转移子采样，在保持10步SDE采样与ODE评估不变的条件下，对比Full-10、Late-5和Uniform-5；完成固定轨迹概率/梯度回归、训练预算与断点恢复验证，并在LIBERO-Object上量化任务成功率与learner计算成本。

**只有实测支持时追加：**

> 在[N]个训练seed、每run[B]组等episode采样预算下，[方法]的ODE成功率为[x]%（Full为[y]%），固定缓存learner更新时间降低[z]%，端到端训练时间变化[q]%。

**负结果也能具体表达：**

> 实测转移子采样使learner耗时降低[z]%，但端到端收益受环境交互占比限制；固定末段与均匀随机策略呈现[性能差异/稳定性差异]，据此明确该优化的适用边界。

面试时应能解释五件事：为什么五步平均除以5；为何Uniform在固定缓存上无偏而Late不是；为什么单epoch的clip不提供实际更新约束；为什么局部加速不等于整次训练加速；如何保证恢复后的reference仍是初始SFT。

## 18. 从实施到结题的一页验收清单

- [ ] 三组方法均在配置与任务清单中，未删减Late-5。
- [ ] 上游commit、SFT子目录/revision及实际权重hash已锁定。
- [ ] 真实GPU/CPU/RAM与EGL可用，原始SFT能够完成有限步ODE评估。
- [ ] 原始Full、共同修复Full和选步接口Full的回归关系成立。
- [ ] state_proj显式冻结，reference来源独立，可训练/实际梯度参数列表可查。
- [ ] 选步只影响learner；五步模式current/ref各调用5次、Full各10次；rollout始终为10步。
- [ ] C1–C11通过，GPU old/current健康检查通过。
- [ ] attempted-budget正常结束和final保存覆盖collapsed边界。
- [ ] 固定缓存三组profile、梯度抽查和全十步漂移记录完成。
- [ ] 三组pilot均完成，或发生的失败已有可定位记录。
- [ ] 正式B与seed矩阵在test之前锁定；机器时间留有评估余量。
- [ ] 最终三张表、逐任务结果、原始日志和限制说明齐全。

**项目的最终目标是交付一个能够被检查和复现的三组对照，并据实回答成本与性能问题。结果方向由实验决定，完成范围由本协议约束。**
