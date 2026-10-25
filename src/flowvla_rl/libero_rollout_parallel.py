"""向量化的 LIBERO 组采集：把 GRPO 一个组的 G 条 episode 并行跑。

**为什么需要它**：实测（scripts/bench_rollout.py）显示单环境 rollout 里
GPU 只占 7.8%，CPU 部分（env_reset 54.3% + 仿真 37.6%）占 91.9%。
按 Amdahl，16 路并行可得约 7.4x 加速——这是整个 RL 阶段成本的头号杠杆。

**与串行版的关系**：`libero_rollout.py` 已通过 34 项测试，保留为**参照 oracle**。
本模块必须与它统计等价（`scripts/compare_rollout.py` 对拍），验证通过后才用于正式实验。

**GRPO 的硬前提**：组内 G 条 episode 必须从**完全相同的初始状态**出发，
否则组均值不是合法 baseline（优势里会混进"状态难易"而非"动作好坏"）。
本模块用与串行版相同的观测哈希断言强制这一点。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from flowvla_rl.grpo import compute_group_advantages
from flowvla_rl.libero_rollout import (
    CAMERA_MAPPING,
    SUITE_MAX_STEPS,
    EpisodeResult,
    LiberoRuntime,
    _success_from_step,
    observation_digest,
)
from flowvla_rl.smolvla_sde import prepare_policy_prefix, sample_sde_chunk
from flowvla_rl.training import StoredChunk, tree_to


@dataclass
class ParallelLiberoRuntime(LiberoRuntime):
    """与串行 runtime 同构，但 env 是 n_envs=G 的 AsyncVectorEnv。"""

    n_envs: int = 1


def make_parallel_runtime(
    checkpoint: str,
    suite: str,
    task_id: int,
    *,
    n_envs: int,
    device: str = "cuda",
) -> ParallelLiberoRuntime:
    """构建 n_envs 路并行的 runtime；策略只加载一份，批量前向。"""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    if n_envs < 2:
        raise ValueError("并行 runtime 至少需要 2 个环境；单环境请用串行版")

    env_config = LiberoEnv(
        task=suite, task_ids=[task_id], camera_name_mapping=CAMERA_MAPPING
    )
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(policy_config, SmolVLAConfig):
        raise TypeError(f"expected a SmolVLA checkpoint, got {type(policy_config).__name__}")
    policy_config.pretrained_path = checkpoint
    policy_config.device = device
    policy_config.empty_cameras = 1
    # 与串行版一致：RL 阶段冻结 VLM，只训 action expert。
    policy_config.train_expert_only = True
    policy_config.freeze_vision_encoder = True
    policy = make_policy(cfg=policy_config, env_cfg=env_config)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_config,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_config, policy_cfg=policy_config
    )
    envs = make_env(env_config, n_envs=n_envs, use_async_envs=True)
    return ParallelLiberoRuntime(
        env=envs[suite][task_id],
        env_config=env_config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        n_envs=n_envs,
    )


def _set_init_state_all(env: Any, init_state_id: int) -> None:
    """把所有子环境设到同一个 init_state_id。

    LeRobot 的 `_LazyAsyncVectorEnv` 只代理了 reset/step/call/get_attr/close，
    **没有 set_attr**。但它内部持有真正的 AsyncVectorEnv，`_ensure()` 之后可直达。
    这是对 v0.6.0 内部结构的依赖，因此下面用 get_attr 回读做强校验——
    一旦上游改名，这里会立刻报错而不是静默使用错误的初始态。
    """
    if hasattr(env, "set_attr"):
        env.set_attr("init_state_id", init_state_id)
    elif hasattr(env, "_ensure"):
        env._ensure()
        inner = getattr(env, "_env", None)
        if inner is None or not hasattr(inner, "set_attr"):
            raise RuntimeError("无法在向量环境上设置 init_state_id：内部对象不支持 set_attr")
        inner.set_attr("init_state_id", init_state_id)
    else:
        raise RuntimeError("向量环境既无 set_attr 也无 _ensure，无法控制初始状态")

    got = list(env.get_attr("init_state_id"))
    if any(int(v) != int(init_state_id) for v in got):
        raise RuntimeError(f"init_state_id 回读不一致：期望 {init_state_id}，实得 {got}")


def _split_observation(obs: dict[str, Any], index: int, n_envs: int) -> dict[str, Any]:
    """从批量观测里取出第 index 个环境的那一份，用于逐环境算哈希。"""
    out: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            out[key] = _split_observation(value, index, n_envs)
            continue
        array = np.asarray(value)
        out[key] = array[index] if array.shape[:1] == (n_envs,) else array
    return out


def assert_identical_initial_states(obs: dict[str, Any], n_envs: int) -> str:
    """GRPO 的硬前提：组内所有 episode 必须从完全相同的初始观测出发。"""
    digests = [observation_digest(_split_observation(obs, i, n_envs)) for i in range(n_envs)]
    if len(set(digests)) != 1:
        raise AssertionError(
            f"GRPO 组内初始观测不一致：{len(set(digests))} 种不同哈希（应为 1 种）"
        )
    return digests[0]


def collect_group_parallel(
    runtime: ParallelLiberoRuntime,
    *,
    eta: float,
    init_state_id: int,
    reset_seed: int,
    policy_seed: int,
    episode_id_start: int = 0,
) -> tuple[list[EpisodeResult], list[StoredChunk]]:
    """并行采集一个 reset-matched 组，返回与串行版同构的结果。

    每个环境独立计步、独立判定终止；已结束的环境不再记录 chunk
    （向量环境仍会继续 step 它们，但其输出被忽略）。
    """
    from lerobot.envs import preprocess_observation
    from lerobot.utils.constants import ACTION

    group_size = runtime.n_envs
    max_steps = SUITE_MAX_STEPS[runtime.env_config.task]
    action_dim = runtime.policy.config.action_feature.shape[0]
    chunk_size = runtime.policy.config.chunk_size

    torch.manual_seed(policy_seed)
    _set_init_state_all(runtime.env, init_state_id)
    observation, _ = runtime.env.reset(seed=reset_seed)
    initial_digest = assert_identical_initial_states(observation, group_size)

    total_steps = np.zeros(group_size, dtype=np.int64)
    finished = np.zeros(group_size, dtype=bool)
    success = np.zeros(group_size, dtype=bool)
    per_env_chunks: list[list[dict[str, Any]]] = [[] for _ in range(group_size)]

    while not finished.all():
        policy_observation = preprocess_observation(observation)
        policy_observation["task"] = list(runtime.env.call("task_description"))
        policy_observation = runtime.env_preprocessor(policy_observation)
        conditioning = runtime.preprocessor(policy_observation)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prefix_masks, prefix_cache = prepare_policy_prefix(runtime.policy, conditioning)
            trajectory = sample_sde_chunk(
                runtime.policy.model,
                prefix_masks,
                prefix_cache,
                action_dim=action_dim,
                eta=eta,
            )
        actions = runtime.postprocessor(trajectory.actions[:, :, :action_dim])
        actions = runtime.env_postprocessor({ACTION: actions})[ACTION]

        valid_positions = torch.zeros((group_size, chunk_size), dtype=torch.bool)
        horizon = min(
            runtime.policy.config.n_action_steps,
            int(max_steps - total_steps[~finished].min()),
        )
        for position in range(horizon):
            action_numpy = actions[:, position].detach().cpu().numpy()
            observation, reward, terminated, truncated, info = runtime.env.step(action_numpy)

            step_success = _per_env_success(reward, terminated, info, group_size)
            newly_active = ~finished
            total_steps[newly_active] += 1
            valid_positions[torch.from_numpy(newly_active), position] = True

            success |= step_success & newly_active
            finished |= (step_success & newly_active) | (total_steps >= max_steps)
            if finished.all():
                break

        # 只为「本 chunk 内至少执行过一步」的环境记录数据
        single = trajectory.to("cpu")
        cond_cpu = tree_to(conditioning, "cpu")
        for index in range(group_size):
            if not bool(valid_positions[index].any()):
                continue
            per_env_chunks[index].append(
                {
                    "conditioning": _index_tree(cond_cpu, index),
                    "trajectory": _index_trajectory(single, index),
                    "valid_positions": valid_positions[index : index + 1],
                }
            )

    episodes = [
        EpisodeResult(
            success=bool(success[i]),
            steps=int(total_steps[i]),
            initial_observation_digest=initial_digest,
            chunks=[],
        )
        for i in range(group_size)
    ]

    rewards = torch.tensor([[float(e.success) for e in episodes]], dtype=torch.float32)
    advantages = compute_group_advantages(rewards)[0]

    stored: list[StoredChunk] = []
    for index, advantage in enumerate(advantages.tolist()):
        for chunk in per_env_chunks[index]:
            stored.append(
                StoredChunk(
                    episode_id=episode_id_start + index,
                    conditioning_batch=chunk["conditioning"],
                    trajectory=chunk["trajectory"],
                    valid_positions=chunk["valid_positions"],
                    advantage=float(advantage),
                )
            )
    return episodes, stored


def _per_env_success(reward: Any, terminated: Any, info: dict[str, Any], n: int) -> np.ndarray:
    """逐环境判定成功，兼容标量/数组/list 三种上报形式。"""
    flags = np.zeros(n, dtype=bool)
    reward_array = np.asarray(reward).reshape(-1)
    if reward_array.size == n:
        flags |= reward_array > 0
    if "is_success" in info:
        value = np.asarray(info["is_success"]).reshape(-1)
        if value.size == n:
            flags |= value.astype(bool)
    if not flags.any():
        for i in range(n):
            flags[i] = _success_from_step(
                np.asarray(reward).reshape(-1)[i] if reward_array.size == n else reward,
                np.asarray(terminated).reshape(-1)[i] if np.asarray(terminated).size == n else terminated,
                {},
            )
    return flags


def _index_tree(node: Any, index: int) -> Any:
    if isinstance(node, dict):
        return {k: _index_tree(v, index) for k, v in node.items()}
    if isinstance(node, torch.Tensor) and node.ndim >= 1:
        return node[index : index + 1]
    if isinstance(node, list):
        return [node[index]]
    return node


def _index_trajectory(trajectory: Any, index: int) -> Any:
    from flowvla_rl.smolvla_sde import SmolVLATrajectory

    return SmolVLATrajectory(
        states=trajectory.states[index : index + 1],
        element_log_probs=trajectory.element_log_probs[index : index + 1],
        valid_action_mask=trajectory.valid_action_mask[index : index + 1],
        eta=trajectory.eta,
    )
