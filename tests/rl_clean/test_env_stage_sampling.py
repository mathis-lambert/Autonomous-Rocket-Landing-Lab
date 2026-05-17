import numpy as np

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.rl import (
    EnvConfig,
    RewardConfig,
    RocketLanderEnv,
    build_default_curriculum,
    find_stage_by_name,
)
from rocket_landing.rl.env import OBSERVATION_SCALE


def test_touchdown_stage_sampling_range() -> None:
    params = RocketParams()
    stages = build_default_curriculum(params)
    stage = find_stage_by_name(stages, "touchdown")
    env = RocketLanderEnv(params, env_config=EnvConfig(), reward_config=RewardConfig())
    env.apply_curriculum_stage(stage)

    for seed in range(8):
        observation, _info = env.reset(seed=seed)
        dx = float(observation[0] * OBSERVATION_SCALE[0])
        z = float(observation[1] * OBSERVATION_SCALE[1])
        assert -0.75 <= dx <= 0.75
        assert 13.0 <= z <= 17.0

    env.close()


def test_normalized_observation_space_contains_reset() -> None:
    params = RocketParams()
    env = RocketLanderEnv(params, env_config=EnvConfig(), reward_config=RewardConfig())
    observation, _info = env.reset(seed=0)
    assert env.observation_space.contains(observation)
    env.close()


def test_terminal_info_includes_truncation_reason() -> None:
    params = RocketParams()
    env = RocketLanderEnv(
        params,
        env_config=EnvConfig(max_episode_steps=1),
        reward_config=RewardConfig(),
    )

    env.reset(seed=0)
    _observation, _reward, _terminated, truncated, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )

    assert truncated
    assert info["truncation_reason"] == "max_episode_steps"
    assert info["episode"]["truncation_reason"] == "max_episode_steps"
    env.close()
