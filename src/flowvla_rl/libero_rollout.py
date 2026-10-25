"""LIBERO group rollout collection for nested control/denoising GRPO."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from flowvla_rl.grpo import compute_group_advantages
from flowvla_rl.smolvla_sde import (
    prepare_policy_prefix,
    rollout_autocast,
    sample_sde_chunk,
)
from flowvla_rl.training import StoredChunk, tree_to


CAMERA_MAPPING = {
    "agentview_image": "camera1",
    "robot0_eye_in_hand_image": "camera2",
}
SUITE_MAX_STEPS = {
    "libero_object": 280,
    "libero_spatial": 280,
    "libero_goal": 300,
}


@dataclass
class LiberoRuntime:
    env: Any                      # 当前激活的环境（单任务时就是唯一的那个）
    env_config: Any
    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any
    # 多任务训练用：套件内所有任务的环境。headline 报的是**套件级**成功率，
    # 所以一条 RL run 必须在套件的多个任务间轮换，产出**一个**能覆盖整套件的策略；
    # 只在单任务上训出来的策略代表不了套件（那会退化成 10 个各自为政的策略）。
    envs: dict[int, Any] | None = None
    task_id: int = 0

    def select_task(self, task_id: int) -> None:
        """切换到套件内的另一个任务；单任务 runtime 上只允许选它自己。"""
        if self.envs is None:
            if task_id != self.task_id:
                raise ValueError(
                    f"这是单任务 runtime（task {self.task_id}），无法切到 task {task_id}；"
                    "多任务训练请用 make_runtime(..., task_ids=[...]) 构建"
                )
            return
        if task_id not in self.envs:
            raise KeyError(f"task {task_id} 不在本 runtime 的任务集合 {sorted(self.envs)} 中")
        self.env = self.envs[task_id]
        self.task_id = task_id

    @property
    def task_ids(self) -> list[int]:
        return sorted(self.envs) if self.envs else [self.task_id]

    def close(self) -> None:
        if self.envs:
            for env in self.envs.values():
                try:
                    env.close()
                except Exception:  # 关一个失败不应挡住其余的
                    pass
        else:
            self.env.close()


@dataclass(frozen=True)
class EpisodeChunk:
    conditioning_batch: dict[str, Any]
    trajectory: Any
    valid_positions: torch.Tensor


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    steps: int
    initial_observation_digest: str
    chunks: list[EpisodeChunk]
    # 原始初始观测：仅用于组内**带容差**的同一性比较（见 collect_group）。
    # 哈希无法做容差比较，而 MuJoCo 的 qacc_warmstart 求解器缓存跨 reset 残留，
    # 会在姿态临界稳定的任务上产生 1e-8 量级的 settle 差异 —— 物理上无关紧要，
    # 但会让逐位相等的 digest 断言炸掉（goal/task2、spatial/task4 实测复现）。
    initial_observation: dict[str, Any] = None


def make_runtime(
    checkpoint: str,
    suite: str,
    task_id: int | None = None,
    device: str = "cuda",
    *,
    task_ids: list[int] | None = None,
) -> LiberoRuntime:
    """Load the exact v0.6.0 policy/environment preprocessing stack.

    传 ``task_id`` 得到单任务 runtime（原有行为，测试依赖它）；
    传 ``task_ids`` 得到多任务 runtime，可用 ``select_task`` 在套件内轮换——
    套件级 headline 需要的是后者。
    """
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    if (task_id is None) == (task_ids is None):
        raise ValueError("task_id 与 task_ids 必须且只能指定一个")
    wanted = [task_id] if task_ids is None else sorted(set(task_ids))
    if not wanted:
        raise ValueError("task_ids 不能为空")

    env_config = LiberoEnv(
        task=suite,
        task_ids=wanted,
        camera_name_mapping=CAMERA_MAPPING,
    )
    policy_config = PreTrainedConfig.from_pretrained(checkpoint)
    if not isinstance(policy_config, SmolVLAConfig):
        raise TypeError(f"expected a SmolVLA checkpoint, got {type(policy_config).__name__}")
    policy_config.pretrained_path = checkpoint
    policy_config.device = device
    policy_config.empty_cameras = 1
    # The released LIBERO checkpoint records its SFT-time full-model settings.
    # Online RL deliberately freezes the VLM and updates only the action expert
    # (plus the action/time projections) to keep the reference affordable.
    policy_config.train_expert_only = True
    policy_config.freeze_vision_encoder = True
    policy = make_policy(cfg=policy_config, env_cfg=env_config)
    policy.model.state_proj.requires_grad_(False)
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
    built = make_env(env_config, n_envs=1, use_async_envs=False)
    per_task = {t: built[suite][t] for t in wanted}
    active = wanted[0]
    return LiberoRuntime(
        env=per_task[active],
        env_config=env_config,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env_preprocessor=env_preprocessor,
        env_postprocessor=env_postprocessor,
        envs=per_task if task_ids is not None else None,
        task_id=active,
    )


def _digest_node(node: Any, digest: Any) -> None:
    if isinstance(node, dict):
        for key in sorted(node):
            digest.update(key.encode())
            _digest_node(node[key], digest)
    else:
        array = np.asarray(node)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())


def _copy_observation(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _copy_observation(value) for key, value in node.items()}
    return np.array(node, copy=True)


# 组同一性的容差。物理状态的求解器抖动实测在 1e-8 量级，1e-3（弧度/米）已放大
# 十万倍仍远低于任务相关尺度；像素允许少量 LSB 级差异（亚像素位移的渲染量化）。
# 超出容差仍**硬性报错** —— 那说明初始态真的不同，组均值基线不再合法。
_STATE_ATOL = 1e-3
_PIXEL_MAX_DELTA = 3.0        # uint8 LSB
_PIXEL_MAX_FRACTION = 0.02    # 允许 2% 像素有 LSB 级差异


def assert_group_identity(episodes: list["EpisodeResult"]) -> None:
    """带容差的组同一性检验：G 条 episode 的初始观测必须在物理意义上相同。"""
    if not episodes:
        raise ValueError("group must contain episodes")
    reference = episodes[0].initial_observation
    for index, episode in enumerate(episodes):
        problems: list[str] = []

        def compare(a: Any, b: Any, path: str) -> None:
            if isinstance(a, dict):
                if not isinstance(b, dict) or a.keys() != b.keys():
                    problems.append(f"{path}: observation keys differ")
                    return
                for key in a:
                    compare(a[key], b[key], f"{path}.{key}" if path else key)
                return
            x = np.asarray(a, dtype=np.float64)
            y = np.asarray(b, dtype=np.float64)
            if x.shape != y.shape:
                problems.append(f"{path}: 形状 {x.shape} vs {y.shape}")
                return
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                problems.append(f"{path}: non-finite observation")
                return
            if x.size == 0:
                return
            delta = np.abs(x - y)
            if "pixel" in path or "image" in path:
                if delta.max() > _PIXEL_MAX_DELTA or (delta > 0).mean() > _PIXEL_MAX_FRACTION:
                    problems.append(
                        f"{path}: 像素 max|Δ|={delta.max():.3g} 差异占比 {(delta > 0).mean():.2%}"
                    )
            elif delta.max() > _STATE_ATOL:
                problems.append(f"{path}: 状态 max|Δ|={delta.max():.3g}")

        compare(reference, episode.initial_observation, "")
        if problems:
            raise AssertionError(
                "GRPO 组内初始观测超出容差（组均值基线不再合法）。"
                f"episode 0 vs {index}:\n  " + "\n  ".join(problems)
            )


def observation_digest(observation: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    _digest_node(observation, digest)
    return digest.hexdigest()


_SOFT_RESET_PATCHED = False


def _install_soft_reset_patch() -> None:
    """给 lerobot 的 LiberoEnv 装上"软 reset"通道。

    背景（实测定位，见 notes/lab.md E14）：LiberoEnv.reset() 每次都做 robosuite
    **hard reset** —— 重建 MjSim 并由**私有 RandomState** 重采 BDDL 摆放；
    ``seed()`` 控制不了它，``set_init_state`` 只恢复 qpos/qvel、恢复不了
    模型级的 fixture 位置。后果：spatial/goal 的部分任务上，同 init_state_id
    的两次 reset 给出**宏观不同**的场景（实测 14% 像素、max|Δ|≈195），
    GRPO 的组同一性前提被打破（object 全部任务恰好无模型级采样，才一直没炸）。

    软 reset 跳过 robosuite 重建：warmstart 清零（settle 起点与 hard reset 后的
    新建 sim 一致）→ set_init_state → num_steps_wait 步 settle → 正常出观测。
    走的仍是 gym 的 reset() 接口，TimeLimit/OrderEnforcing 等包装器照常工作；
    官方评测器从不设 ``_flowvla_soft_reset``，行为不受影响。
    """
    global _SOFT_RESET_PATCHED
    if _SOFT_RESET_PATCHED:
        return
    import gymnasium as gym
    from lerobot.envs.libero import LiberoEnv, get_libero_dummy_action

    original_reset = LiberoEnv.reset

    def patched_reset(self, seed=None, **kwargs):
        if not getattr(self, "_flowvla_soft_reset", False):
            return original_reset(self, seed=seed, **kwargs)
        # 快照恢复：把第 1 条 episode settle **完成后**抓下的完整 sim 状态直接
        # set 回去。不重演 settle（重演不可复现：settle 轨迹受控制器目标、
        # 求解器缓存等隐状态影响，实测五种"让重演确定"的尝试全部失败），
        # 不触发 robosuite 场景重采（模型未重建）。
        # set_init_state 本质就是 set_state + forward + 刷新观测，接受任意状态向量。
        gym.Env.reset(self, seed=seed)
        snapshot = getattr(self, "_flowvla_group_snapshot", None)
        if snapshot is None:
            raise RuntimeError("软 reset 需要先有组首条 episode 的状态快照")
        raw_obs = self._env.set_init_state(snapshot)
        # robosuite 的回合簿记也要归零：软路径不走它的 reset，
        # done 标志会残留（下一步直接抛 "executing action in terminated episode"），
        # timestep 会跨 episode 累积、撞上其内部 horizon 后把 done 置 True。
        suite_env = self._env.env
        suite_env.done = False
        suite_env.timestep = 0
        if hasattr(suite_env, "cur_time"):
            suite_env.cur_time = 0
        for robot in self._env.robots:
            controller = robot.controller
            # 控制器缓存的运动学（ee_ori_mat 等）停留在上一条 episode 末尾，
            # 而 eef 观测读的是这份缓存 —— 像素已一致但 eef.mat 差 0.8 就是它。
            # update(force=True) 让缓存从 sim 当前状态重算，再把目标钉到当前位姿。
            if hasattr(controller, "update"):
                try:
                    controller.update(force=True)
                except TypeError:
                    controller.update()
            if hasattr(controller, "reset_goal"):
                controller.reset_goal()
            controller.use_delta = self.control_mode == "relative"
        # 缓存刷新后重新出观测（set_init_state 里那次 _get_observations 读到的还是旧缓存）
        raw_obs = self._env.env._get_observations(force_update=True) if hasattr(
            self._env, "env") else raw_obs
        return self._format_raw_obs(raw_obs), {}

    LiberoEnv.reset = patched_reset
    _SOFT_RESET_PATCHED = True


def _reset_to_initial_state(
    runtime: LiberoRuntime, init_state_id: int, seed: int, *, soft: bool = False
):
    """Reset to an explicit init-state index; seed alone is not sufficient.

    ``soft=True`` 用于组内第 2..G 条 episode：跳过 robosuite 场景重建，
    保证与第 1 条完全同场景（GRPO 组均值基线的前提）。
    """
    if not hasattr(runtime.env, "envs") or len(runtime.env.envs) != 1:
        raise TypeError("group collection requires a one-env SyncVectorEnv")
    _install_soft_reset_patch()
    base_env = runtime.env.envs[0].unwrapped
    base_env.init_state_id = init_state_id
    base_env._flowvla_soft_reset = bool(soft)
    observation, info = runtime.env.reset(seed=seed)
    if not soft:
        # 组首条：settle 完成后的完整状态快照，供同组后续 episode 恢复
        base_env._flowvla_group_snapshot = base_env._env.sim.get_state().flatten()
        # 观测统一走快照渲染路径：full reset 的 step-渲染与 set_state-渲染有
        # 小幅像素差（实测 max≈47），而组内一致性要求所有 episode 的初始观测
        # 出自同一条渲染管线。物理状态不变（用的就是刚抓的快照），只是重新出观测。
        base_env._flowvla_soft_reset = True
        observation, info = runtime.env.reset(seed=seed)
    # flag **保持 True** 直到下一次显式 full reset：episode 成功结束时
    # SyncVectorEnv 会自动 reset，若走 full 路径会重建模型、重采场景，
    # 让下一条 episode 的快照贴到不同的模型上（组同一性再次被破坏——
    # 手工验证与 collect_group 结果不一致的根因就在这，手工序列没触发过 done）。
    base_env._flowvla_soft_reset = True
    return observation, info


def _success_from_step(reward: Any, terminated: Any, info: dict[str, Any]) -> bool:
    if bool(np.asarray(reward).any()):
        return True
    if "is_success" in info and bool(np.asarray(info["is_success"]).any()):
        return True
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            return bool(np.asarray(final_info.get("is_success", False)).any())
        return any(
            isinstance(item, dict) and bool(item.get("is_success", False))
            for item in np.asarray(final_info, dtype=object).reshape(-1)
        )
    return bool(np.asarray(terminated).any()) and bool(np.asarray(reward).any())


def collect_episode(
    runtime: LiberoRuntime,
    *,
    eta: float,
    init_state_id: int,
    reset_seed: int,
    policy_seed: int,
    soft_reset: bool = False,
) -> EpisodeResult:
    """Collect one complete episode while retaining only chunk-level RL data."""
    from lerobot.envs import preprocess_observation
    from lerobot.utils.constants import ACTION

    torch.manual_seed(policy_seed)
    runtime.policy.eval()
    observation, _ = _reset_to_initial_state(
        runtime, init_state_id, reset_seed, soft=soft_reset
    )
    initial_digest = observation_digest(observation)
    observation_snapshot = _copy_observation(observation)
    chunks: list[EpisodeChunk] = []
    total_steps = 0
    success = False
    done = False
    max_steps = SUITE_MAX_STEPS[runtime.env_config.task]
    action_dim = runtime.policy.config.action_feature.shape[0]

    while total_steps < max_steps and not done:
        policy_observation = preprocess_observation(observation)
        policy_observation["task"] = list(runtime.env.call("task_description"))
        policy_observation = runtime.env_preprocessor(policy_observation)
        conditioning = runtime.preprocessor(policy_observation)
        # 与 training.optimize_grpo 的打分共用 rollout_autocast —— 精度必须一致，
        # 否则同一条轨迹算出的 log-prob 不同，ratio 偏离 1，PPO 目标退化成噪声。
        device = next(runtime.policy.parameters()).device
        with torch.no_grad(), rollout_autocast(device):
            prefix_masks, prefix_cache = prepare_policy_prefix(runtime.policy, conditioning)
            trajectory = sample_sde_chunk(
                runtime.policy.model,
                prefix_masks,
                prefix_cache,
                action_dim=action_dim,
                eta=eta,
            )
        normalized_actions = trajectory.actions[:, :, :action_dim]
        actions = runtime.postprocessor(normalized_actions)
        actions = runtime.env_postprocessor({ACTION: actions})[ACTION]
        valid_positions = torch.zeros(
            (1, runtime.policy.config.chunk_size), dtype=torch.bool
        )

        execution_horizon = min(
            runtime.policy.config.n_action_steps,
            max_steps - total_steps,
        )
        for position in range(execution_horizon):
            action_numpy = actions[:, position].detach().cpu().numpy()
            observation, reward, terminated, truncated, info = runtime.env.step(action_numpy)
            valid_positions[0, position] = True
            total_steps += 1
            if hasattr(runtime, "collection_progress"):
                runtime.collection_progress["env_steps"] += 1
            success = _success_from_step(reward, terminated, info)
            done = success or bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any())
            if done:
                break

        chunks.append(
            EpisodeChunk(
                conditioning_batch=tree_to(conditioning, "cpu"),
                trajectory=trajectory.to("cpu"),
                valid_positions=valid_positions,
            )
        )
    if hasattr(runtime, "collection_progress"):
        runtime.collection_progress["completed_episodes"] += 1
    return EpisodeResult(
        success=success,
        steps=total_steps,
        initial_observation_digest=initial_digest,
        chunks=chunks,
        initial_observation=observation_snapshot,
    )


def collect_group(
    runtime: LiberoRuntime,
    *,
    group_size: int,
    eta: float,
    init_state_id: int,
    reset_seed: int,
    policy_seed: int,
    episode_id_start: int = 0,
    chunk_discount: float = 1.0,
) -> tuple[list[EpisodeResult], list[StoredChunk]]:
    """Collect a reset-matched group and broadcast its episode advantages.

    ``chunk_discount``（γ≤1）把终局奖励按 reward-to-go 折到各 chunk：
    第 c 个 chunk（共 C 个）的优势 = A · γ^(C-1-c)，越靠近终局 credit 越大。
    默认 1.0 = 原行为（整条 episode 共享同一个标量优势）。
    动机：1-bit 终局奖励摊到 30–60 个动作上，credit assignment 极粗——
    成功 episode 里早期的徘徊动作与临门的关键动作拿到完全相同的正优势，
    梯度里混入了对无关行为的强化。γ<1 是不引入价值函数的最便宜修正。
    """
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    if not 0.0 < chunk_discount <= 1.0:
        raise ValueError("chunk_discount must lie in (0, 1]")
    episodes = [
        collect_episode(
            runtime,
            eta=eta,
            init_state_id=init_state_id,
            reset_seed=reset_seed,
            policy_seed=policy_seed + index,
            # 第 1 条 full reset（初始态分布与评测协议一致），
            # 其余 soft reset（跳过场景重采，保证组内同场景）
            soft_reset=index > 0,
        )
        for index in range(group_size)
    ]
    # Validate even equal digests: identical NaNs must not pass the identity gate.
    assert_group_identity(episodes)
    rewards = torch.tensor(
        [[float(episode.success) for episode in episodes]], dtype=torch.float32
    )
    advantages = compute_group_advantages(rewards)[0]
    stored: list[StoredChunk] = []
    for index, (episode, advantage) in enumerate(
        zip(episodes, advantages.tolist(), strict=True)
    ):
        chunk_count = len(episode.chunks)
        for position, chunk in enumerate(episode.chunks):
            # reward-to-go：终局奖励对第 position 个 chunk 的折扣是 γ^(距终局的 chunk 数)
            credit = chunk_discount ** (chunk_count - 1 - position)
            stored.append(
                StoredChunk(
                    episode_id=episode_id_start + index,
                    conditioning_batch=chunk.conditioning_batch,
                    trajectory=chunk.trajectory,
                    valid_positions=chunk.valid_positions,
                    advantage=float(advantage) * credit,
                    chunk_id=position,
                )
            )
    return episodes, stored
