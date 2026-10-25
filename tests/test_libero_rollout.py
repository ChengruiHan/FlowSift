"""Group collection tests using tiny stand-ins for the LIBERO/lerobot stack.

Nothing here needs a GPU, lerobot, MuJoCo or LIBERO. The simulator is replaced
by a fake that reproduces the two behaviours the collector actually depends on:
a hard reset re-samples the BDDL fixture layout, and observations rendered from
a restored state differ from the ones a hard reset returns.
"""

import copy
import sys
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from torch import nn

from flowvla_rl import libero_rollout
from flowvla_rl.libero_rollout import (
    EpisodeResult,
    LiberoRuntime,
    _reset_to_initial_state,
    _success_from_step,
    assert_group_identity,
    collect_episode,
    collect_group,
    observation_digest,
)


SUITE = "libero_object"
MAX_STEPS = 10
CHUNK_SIZE = 6
N_ACTION_STEPS = 4
ACTION_DIM = 7
MAX_ACTION_DIM = 8
STATE_DIM = 9
IMAGE_SHAPE = (12, 12, 3)
# A full reset renders through the step pipeline, a restored state through
# set_state; the two disagree by tens of levels (measured max is about 47).
HARD_RESET_PIXEL_BIAS = 47
# set_init_state reports observations read from the pre-refresh controller cache.
STALE_CACHE_MARKER = 5.0


# --------------------------------------------------------------------------- #
# Tolerance-aware group identity
# --------------------------------------------------------------------------- #


def _make_observation(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "pixels": {
            "camera1": rng.integers(0, 200, size=(1, *IMAGE_SHAPE), dtype=np.uint8)
        },
        "agent_pos": rng.normal(size=(1, STATE_DIM)),
    }


def _episode(observation: dict, *, success: bool = False) -> EpisodeResult:
    return EpisodeResult(
        success=success,
        steps=MAX_STEPS,
        initial_observation_digest=observation_digest(observation),
        chunks=[],
        initial_observation=observation,
    )


def test_group_identity_tolerates_solver_jitter_and_lsb_pixel_noise() -> None:
    reference = _make_observation(0)
    jittered = copy.deepcopy(reference)
    # MuJoCo's warmstart cache survives a reset and leaves settle differences
    # around 1e-8; physically irrelevant, but fatal to a bit-exact digest.
    jittered["agent_pos"] += 1e-8
    # Sub-pixel motion quantises to single-LSB differences on a few pixels.
    flat = jittered["pixels"]["camera1"].reshape(-1)
    touched = np.random.default_rng(7).choice(flat.size, size=4, replace=False)
    flat[touched] += 1

    episodes = [_episode(reference), _episode(jittered)]
    # A hash cannot express a tolerance, which is precisely why the digest alone
    # must not gate group identity.
    assert (
        episodes[0].initial_observation_digest
        != episodes[1].initial_observation_digest
    )
    assert_group_identity(episodes)


def test_group_identity_rejects_a_resampled_scene() -> None:
    reference = _make_observation(1)
    resampled = copy.deepcopy(reference)
    # A hard reset re-sampled the fixture layout: a third of the frame moves by
    # far more than a rendering LSB, so the group-mean baseline is invalid.
    resampled["pixels"]["camera1"][:, :4] = np.uint8(255)
    with pytest.raises(AssertionError, match="pixels.camera1"):
        assert_group_identity([_episode(reference), _episode(resampled)])


def test_group_identity_rejects_state_and_shape_mismatches() -> None:
    reference = _make_observation(2)

    drifted = copy.deepcopy(reference)
    drifted["agent_pos"][0, 3] += 1e-2  # ten times the state tolerance
    with pytest.raises(AssertionError, match="agent_pos"):
        assert_group_identity([_episode(reference), _episode(drifted)])

    reshaped = copy.deepcopy(reference)
    reshaped["agent_pos"] = np.zeros((1, STATE_DIM + 1))
    with pytest.raises(AssertionError, match="agent_pos"):
        assert_group_identity([_episode(reference), _episode(reshaped)])


# --------------------------------------------------------------------------- #
# Success normalisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("reward", "terminated", "info", "expected"),
    [
        # Per-env reward vector from a SyncVectorEnv.
        (np.array([1.0]), np.array([False]), {}, True),
        (np.array([0.0]), np.array([False]), {}, False),
        # Explicit per-env success flag.
        (np.array([0.0]), np.array([True]), {"is_success": np.array([True])}, True),
        (np.array([0.0]), np.array([True]), {"is_success": np.array([False])}, False),
        # Autoreset carries the terminal flags in final_info as a dict of arrays.
        (
            np.array([0.0]),
            np.array([True]),
            {"final_info": {"is_success": np.array([True])}},
            True,
        ),
        (
            np.array([0.0]),
            np.array([True]),
            {"final_info": {"is_success": np.array([False])}},
            False,
        ),
        # Older vector envs report final_info as one dict per sub-env.
        (
            np.array([0.0]),
            np.array([True]),
            {"final_info": np.array([{"is_success": True}], dtype=object)},
            True,
        ),
        (
            np.array([0.0]),
            np.array([True]),
            {"final_info": np.array([{"is_success": False}], dtype=object)},
            False,
        ),
        # Scalar reporting from a plain (non-vector) env.
        (1.0, False, {}, True),
        # Termination without reward is a failure, not a success.
        (0.0, True, {}, False),
    ],
)
def test_success_detection_normalises_every_reporting_shape(
    reward, terminated, info, expected
) -> None:
    assert _success_from_step(reward, terminated, info) is expected


# --------------------------------------------------------------------------- #
# Tiny stand-ins for the policy and the simulator
# --------------------------------------------------------------------------- #


class TinyFlowModel(nn.Module):
    """Minimal VLAFlowMatching stand-in with the same call surface."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            max_action_dim=MAX_ACTION_DIM,
            num_steps=2,
            use_cache=True,
        )
        self.vlm_with_expert = SimpleNamespace(
            forward=lambda **kwargs: (None, "prefix-cache")
        )

    def embed_prefix(self, images, image_masks, language_tokens, language_masks, state):
        del images, image_masks, language_tokens, language_masks
        batch = state.shape[0]
        return (
            torch.zeros(batch, 3, 2),
            torch.ones(batch, 3, dtype=torch.bool),
            torch.zeros(batch, 3, dtype=torch.long),
        )

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    def denoise_step(self, *, x_t, prefix_pad_masks, past_key_values, timestep):
        del prefix_pad_masks, past_key_values
        return 0.1 * x_t + timestep[:, None, None]


class TinyPolicy(nn.Module):
    """Minimal SmolVLAPolicy stand-in, driven through prepare_policy_prefix."""

    def __init__(self) -> None:
        super().__init__()
        self.expert_scale = nn.Parameter(torch.ones(1))
        self.model = TinyFlowModel()
        self.config = SimpleNamespace(
            chunk_size=CHUNK_SIZE,
            n_action_steps=N_ACTION_STEPS,
            action_feature=SimpleNamespace(shape=(ACTION_DIM,)),
        )

    def _prepare_batch(self, batch):
        return batch

    def prepare_images(self, batch):
        image = batch["observation.images.camera1"]
        return [image], [torch.ones(image.shape[0], dtype=torch.bool)]

    def prepare_state(self, batch):
        return batch["observation.state"]


def fake_preprocessor(observation: dict) -> dict:
    """Stand-in for the SmolVLA preprocessor: emit the keys the prefix needs."""
    if "task" not in observation:
        raise KeyError("the task description must be attached before preprocessing")
    state = torch.as_tensor(np.asarray(observation["agent_pos"]), dtype=torch.float32)
    image = torch.as_tensor(
        np.asarray(observation["pixels"]["camera1"]), dtype=torch.float32
    )
    batch = state.shape[0]
    return {
        "observation.state": state,
        "observation.images.camera1": image,
        "observation.language.tokens": torch.zeros(batch, 3, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(batch, 3, dtype=torch.bool),
    }


class FakeController:
    """robosuite controller whose kinematics cache the soft path must refresh."""

    def __init__(self) -> None:
        self.update_calls: list[bool] = []
        self.goal_resets = 0
        self.use_delta = None

    def update(self, force=False):
        self.update_calls.append(bool(force))

    def reset_goal(self):
        self.goal_resets += 1


class FakeRobosuiteEnv:
    """Episode bookkeeping that a soft reset has to zero out by hand."""

    def __init__(self, scene: "FakeScene") -> None:
        self._scene = scene
        self.done = True
        self.timestep = 137
        self.cur_time = 4.25
        self.observation_calls = 0

    def _get_observations(self, force_update=False):
        del force_update
        self.observation_calls += 1
        return self._scene.render()


class FakeScene:
    """The MuJoCo side: a re-samplable fixture layout plus a state vector."""

    def __init__(self) -> None:
        self.layout = 0
        self.qpos = np.zeros(STATE_DIM)
        self.env = FakeRobosuiteEnv(self)
        self.robots = [SimpleNamespace(controller=FakeController())]
        self.sim = SimpleNamespace(
            get_state=lambda: SimpleNamespace(flatten=lambda: self.qpos.copy())
        )
        self.set_init_state_calls = 0

    def render(self, *, marker: float = 0.0, pixel_bias: int = 0) -> dict:
        image = np.full(IMAGE_SHAPE, 10, dtype=np.uint8)
        # The BDDL fixture block, re-sampled by every hard reset.
        image[:4] = np.uint8(20 + 70 * self.layout)
        # The robot, which moves with the state vector.
        image[4:8] = np.uint8(int(abs(self.qpos[0]) * 40.0) % 200)
        if pixel_bias:
            image = np.clip(image.astype(np.int16) + pixel_bias, 0, 255).astype(np.uint8)
        return {"agentview_image": image, "robot0_eef_pos": self.qpos + marker}

    def set_init_state(self, state):
        self.set_init_state_calls += 1
        self.qpos = np.asarray(state, dtype=np.float64).copy()
        # Mirrors the real call: the observation still comes from the stale
        # controller cache, which is why the patch re-renders afterwards.
        return self.render(marker=STALE_CACHE_MARKER)


class FakeLiberoEnv(gym.Env):
    """lerobot's LiberoEnv, including the hard reset that re-samples the scene."""

    def __init__(self) -> None:
        self._env = FakeScene()
        self.control_mode = "relative"
        self.init_state_id = None
        self.hard_resets = 0

    def reset(self, seed=None, **kwargs):
        del kwargs
        gym.Env.reset(self, seed=seed)
        self.hard_resets += 1
        # robosuite rebuilds MjSim and re-samples the layout from a private RNG
        # that neither `seed` nor `init_state_id` reaches. This is the bug the
        # snapshot soft reset exists to work around.
        self._env.layout = self.hard_resets
        self._env.qpos = np.full(STATE_DIM, float(self.init_state_id or 0) + 0.5)
        self._env.env.done = False
        self._env.env.timestep = 0
        raw_obs = self._env.render(pixel_bias=HARD_RESET_PIXEL_BIAS)
        return self._format_raw_obs(raw_obs), {}

    def step(self, action):
        self._env.qpos = self._env.qpos + 0.05 * float(np.asarray(action).sum())
        self._env.env.timestep += 1
        return self._format_raw_obs(self._env.render()), 0.0, False, False, {}

    def _format_raw_obs(self, raw_obs):
        return {
            "pixels": {"camera1": raw_obs["agentview_image"]},
            "agent_pos": raw_obs["robot0_eef_pos"],
        }


def _batch(observation: dict) -> dict:
    return {
        "pixels": {name: image[None] for name, image in observation["pixels"].items()},
        "agent_pos": observation["agent_pos"][None],
    }


class FakeSyncVectorEnv:
    """One-env stand-in for the SyncVectorEnv group collection requires."""

    def __init__(self, base_env: FakeLiberoEnv, *, success_at=()) -> None:
        self._base = base_env
        self.envs = [SimpleNamespace(unwrapped=base_env)]
        self.success_at = list(success_at)
        self.episode_index = -1
        self.step_count = 0
        self.total_steps = 0
        self._fresh = True

    def reset(self, seed=None):
        observation, info = self._base.reset(seed=seed)
        self.step_count = 0
        self._fresh = True
        return _batch(observation), info

    def call(self, name):
        assert name == "task_description"
        return ("pick up the black bowl",)

    def step(self, action):
        # Episodes are counted from the first step after a reset, so the extra
        # snapshot reset the collector performs does not shift the schedule.
        if self._fresh:
            self._fresh = False
            self.episode_index += 1
        observation, *_ = self._base.step(np.asarray(action)[0])
        self.step_count += 1
        self.total_steps += 1
        target = (
            self.success_at[self.episode_index]
            if self.episode_index < len(self.success_at)
            else None
        )
        success = target is not None and self.step_count >= target
        return (
            _batch(observation),
            np.array([1.0 if success else 0.0], dtype=np.float32),
            np.array([success]),
            np.array([False]),
            {"is_success": np.array([success])},
        )


def _make_runtime(*, success_at=()) -> LiberoRuntime:
    return LiberoRuntime(
        env=FakeSyncVectorEnv(FakeLiberoEnv(), success_at=success_at),
        env_config=SimpleNamespace(task=SUITE),
        policy=TinyPolicy(),
        preprocessor=fake_preprocessor,
        postprocessor=lambda actions: actions * 2.0,
        env_preprocessor=lambda observation: observation,
        env_postprocessor=lambda batch: batch,
    )


@pytest.fixture
def libero_stack(monkeypatch):
    """Install the lerobot symbols the collector imports lazily."""
    lerobot = ModuleType("lerobot")
    envs = ModuleType("lerobot.envs")
    envs.preprocess_observation = lambda observation: dict(observation)
    libero = ModuleType("lerobot.envs.libero")
    libero.LiberoEnv = FakeLiberoEnv
    libero.get_libero_dummy_action = lambda: np.zeros(ACTION_DIM)
    utils = ModuleType("lerobot.utils")
    constants = ModuleType("lerobot.utils.constants")
    constants.ACTION = "action"
    lerobot.envs = envs
    lerobot.utils = utils
    envs.libero = libero
    utils.constants = constants
    for name, module in (
        ("lerobot", lerobot),
        ("lerobot.envs", envs),
        ("lerobot.envs.libero", libero),
        ("lerobot.utils", utils),
        ("lerobot.utils.constants", constants),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # The soft-reset patch is installed once per process onto LiberoEnv.reset;
    # restore both the flag and the class attribute so tests stay independent.
    monkeypatch.setattr(libero_rollout, "_SOFT_RESET_PATCHED", False)
    monkeypatch.setattr(FakeLiberoEnv, "reset", FakeLiberoEnv.reset)
    monkeypatch.setitem(libero_rollout.SUITE_MAX_STEPS, SUITE, MAX_STEPS)


# --------------------------------------------------------------------------- #
# Snapshot soft reset
# --------------------------------------------------------------------------- #


def test_soft_reset_restores_the_snapshot_without_resampling_the_scene(
    libero_stack,
) -> None:
    runtime = _make_runtime()
    base_env = runtime.env.envs[0].unwrapped

    first, _ = _reset_to_initial_state(runtime, init_state_id=3, seed=11, soft=False)
    assert base_env.init_state_id == 3
    assert base_env.hard_resets == 1
    snapshot = base_env._flowvla_group_snapshot.copy()
    # The group's first episode is re-rendered through the snapshot path, so its
    # observation carries neither the hard-reset render bias nor a stale marker.
    np.testing.assert_array_equal(first["agent_pos"][0], snapshot)
    assert not np.array_equal(
        first["pixels"]["camera1"][0],
        base_env._env.render(pixel_bias=HARD_RESET_PIXEL_BIAS)["agentview_image"],
    )
    np.testing.assert_array_equal(
        first["pixels"]["camera1"][0], base_env._env.render()["agentview_image"]
    )
    # The flag stays on so an autoreset after a terminal step cannot rebuild the
    # model behind the collector's back.
    assert base_env._flowvla_soft_reset is True

    # Drive the simulator away from the snapshot and dirty robosuite's state.
    for _ in range(5):
        runtime.env.step(np.ones((1, ACTION_DIM)))
    assert not np.allclose(base_env._env.qpos, snapshot)
    base_env._env.env.done = True
    base_env._env.env.timestep = 42
    base_env._env.env.cur_time = 9.5
    controller = base_env._env.robots[0].controller
    controller.update_calls.clear()
    controller.goal_resets = 0

    second, _ = _reset_to_initial_state(runtime, init_state_id=3, seed=11, soft=True)

    # No fresh scene sample: the whole group sees one and the same layout.
    assert base_env.hard_resets == 1
    assert base_env._env.set_init_state_calls == 2
    np.testing.assert_array_equal(base_env._env.qpos, snapshot)
    # robosuite bookkeeping is cleared by hand, since the soft path skips reset.
    assert base_env._env.env.done is False
    assert base_env._env.env.timestep == 0
    assert base_env._env.env.cur_time == 0
    # The controller cache is recomputed from the restored sim state.
    assert controller.update_calls == [True]
    assert controller.goal_resets == 1
    assert controller.use_delta is True
    # Observations are re-rendered after that refresh, so they match the first
    # episode exactly instead of carrying set_init_state's stale values.
    np.testing.assert_array_equal(second["agent_pos"][0], snapshot)
    np.testing.assert_array_equal(
        second["pixels"]["camera1"], first["pixels"]["camera1"]
    )


def test_soft_reset_rejects_a_missing_snapshot_and_a_multi_env_runtime(
    libero_stack,
) -> None:
    runtime = _make_runtime()
    # Nothing has been captured yet, so there is no group state to restore.
    with pytest.raises(RuntimeError):
        _reset_to_initial_state(runtime, init_state_id=0, seed=0, soft=True)

    runtime.env.envs = runtime.env.envs * 2
    with pytest.raises(TypeError, match="one-env"):
        _reset_to_initial_state(runtime, init_state_id=0, seed=0, soft=False)


# --------------------------------------------------------------------------- #
# Episode collection
# --------------------------------------------------------------------------- #


def test_step_cap_is_enforced_across_partial_chunks(libero_stack) -> None:
    runtime = _make_runtime()
    result = collect_episode(
        runtime, eta=0.4, init_state_id=0, reset_seed=1, policy_seed=0
    )

    assert result.success is False
    assert result.steps == MAX_STEPS
    assert runtime.env.total_steps == MAX_STEPS
    # 10 steps at 4 executed actions per chunk: the tail chunk is truncated by
    # the cap rather than running past it.
    assert len(result.chunks) == 3
    assert [int(chunk.valid_positions.sum()) for chunk in result.chunks] == [4, 4, 2]
    assert all(
        chunk.valid_positions.shape == (1, CHUNK_SIZE) for chunk in result.chunks
    )
    assert result.chunks[-1].valid_positions.tolist() == [
        [True, True, False, False, False, False]
    ]
    assert result.chunks[0].trajectory.states.shape == (
        1,
        runtime.policy.model.config.num_steps + 1,
        CHUNK_SIZE,
        MAX_ACTION_DIM,
    )


def test_episode_stops_at_the_first_success_signal(libero_stack) -> None:
    runtime = _make_runtime(success_at=(3,))
    result = collect_episode(
        runtime, eta=0.4, init_state_id=0, reset_seed=1, policy_seed=0
    )

    assert result.success is True
    assert result.steps == 3
    # The env is not stepped again once success is reported.
    assert runtime.env.total_steps == 3
    assert len(result.chunks) == 1
    assert result.chunks[0].valid_positions.tolist() == [
        [True, True, True, False, False, False]
    ]


# --------------------------------------------------------------------------- #
# Group collection
# --------------------------------------------------------------------------- #


def test_collect_group_shares_one_scene_and_broadcasts_advantages(
    libero_stack,
) -> None:
    runtime = _make_runtime(success_at=(None, 3, None))
    episodes, stored = collect_group(
        runtime,
        group_size=3,
        eta=0.4,
        init_state_id=2,
        reset_seed=5,
        policy_seed=0,
        episode_id_start=10,
    )
    base_env = runtime.env.envs[0].unwrapped

    # One hard reset for the whole group: members 2..G restore the snapshot.
    assert base_env.hard_resets == 1
    assert len({episode.initial_observation_digest for episode in episodes}) == 1
    assert_group_identity(episodes)  # the tolerant path agrees with the digest

    assert [episode.success for episode in episodes] == [False, True, False]
    assert [episode.steps for episode in episodes] == [MAX_STEPS, 3, MAX_STEPS]
    assert [len(episode.chunks) for episode in episodes] == [3, 1, 3]

    # One scalar Dr.GRPO advantage per episode, broadcast over its chunks.
    assert [chunk.episode_id for chunk in stored] == [10, 10, 10, 11, 12, 12, 12]
    assert [chunk.advantage for chunk in stored] == pytest.approx(
        [-1 / 3] * 3 + [2 / 3] + [-1 / 3] * 3
    )
    assert sum(chunk.advantage for chunk in stored) != pytest.approx(0.0)


def test_chunk_discount_applies_reward_to_go_credit(libero_stack) -> None:
    runtime = _make_runtime(success_at=(None, 3))
    episodes, stored = collect_group(
        runtime,
        group_size=2,
        eta=0.4,
        init_state_id=1,
        reset_seed=3,
        policy_seed=0,
        chunk_discount=0.5,
    )

    assert [len(episode.chunks) for episode in episodes] == [3, 1]
    # Chunk c of C gets A * gamma ** (C - 1 - c): the closer to the terminal
    # reward, the more credit.
    assert [chunk.advantage for chunk in stored] == pytest.approx(
        [-0.5 * 0.25, -0.5 * 0.5, -0.5, 0.5]
    )

    with pytest.raises(ValueError, match="group_size"):
        collect_group(
            runtime, group_size=1, eta=0.4, init_state_id=1, reset_seed=3, policy_seed=0
        )
    for invalid in (0.0, 1.5):
        with pytest.raises(ValueError, match="chunk_discount"):
            collect_group(
                runtime,
                group_size=2,
                eta=0.4,
                init_state_id=1,
                reset_seed=3,
                policy_seed=0,
                chunk_discount=invalid,
            )

@pytest.mark.parametrize('terminal_kind', ['terminated', 'truncated'])
def test_early_unsuccessful_end_stops_episode(libero_stack, terminal_kind):
    runtime = _make_runtime()
    original_step = runtime.env.step

    def step(action):
        observation, reward, terminated, truncated, info = original_step(action)
        if runtime.env.total_steps > 2:
            pytest.fail('collector stepped past an unsuccessful terminal transition')
        if runtime.env.total_steps == 2:
            if terminal_kind == 'terminated':
                terminated[:] = True
            else:
                truncated[:] = True
        return observation, reward, terminated, truncated, info

    runtime.env.step = step
    episode = collect_episode(runtime, eta=0.4, init_state_id=0, reset_seed=0, policy_seed=7)
    assert not episode.success
    assert episode.steps == 2
    assert len(episode.chunks) == 1
    assert episode.chunks[0].valid_positions.sum().item() == 2


@pytest.mark.parametrize('bad_value', [float('nan'), float('inf')])
def test_identity_rejects_identical_nonfinite_observations(bad_value):
    obs = _make_observation(0)
    obs['agent_pos'][0, 0] = bad_value
    with pytest.raises(AssertionError, match='non-finite'):
        assert_group_identity([_episode(obs), _episode(copy.deepcopy(obs))])


def test_identity_rejects_missing_keys():
    obs = _make_observation(0)
    changed = copy.deepcopy(obs)
    del changed['agent_pos']
    with pytest.raises(AssertionError, match='keys differ'):
        assert_group_identity([_episode(obs), _episode(changed)])
