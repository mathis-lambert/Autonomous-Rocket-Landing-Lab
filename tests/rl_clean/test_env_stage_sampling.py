import numpy as np

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.collision import GroundContactResolver
from rocket_landing.rl import (
    EnvConfig,
    RewardConfig,
    RocketLanderEnv,
    build_default_curriculum,
    find_stage_by_name,
)
from rocket_landing.rl.env import OBSERVATION_SCALE, build_observation


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


def test_observation_is_clipped_to_declared_box() -> None:
    params = RocketParams()
    env = RocketLanderEnv(params, env_config=EnvConfig(), reward_config=RewardConfig())

    raw_state = State(
        x=params.target_x + 500.0,
        z=800.0,
        vx=200.0,
        vz=-200.0,
        theta=0.0,
        omega=20.0,
        fuel=params.initial_fuel * 2.0,
    )
    encoded = build_observation(raw_state, params)

    assert env.observation_space.contains(encoded)
    assert np.all(encoded <= 1.0)
    assert np.all(encoded >= -1.0)
    env.close()


def test_soft_landing_uses_wrapped_attitude_error() -> None:
    params = RocketParams()
    resolver = GroundContactResolver(params)
    grounded_state = State(
        x=params.target_x,
        z=params.length * 0.5,
        vx=0.0,
        vz=0.0,
        theta=(2.0 * np.pi) + (0.5 * params.max_landing_theta),
        omega=0.0,
        fuel=params.initial_fuel,
    )

    assert resolver._is_soft_landing(grounded_state)
