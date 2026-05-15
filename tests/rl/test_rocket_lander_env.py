import numpy as np
import pytest

from rocket_landing.application.rl.config import RLEnvConfig
from rocket_landing.application.rl.curriculum.stages import CurriculumStage, PromotionCriteria
from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.application.rl.reset.distributions import InitialStateDistribution, UniformRange
from rocket_landing.application.rl.termination import TruncationConfig
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


def test_env_reset_returns_expected_observation_shape() -> None:
    env = RocketLanderEnv(RocketParams())

    observation, info = env.reset(seed=7)

    assert observation.shape == (8,)
    assert info["fuel_ratio"] <= 1.0


def test_env_step_requires_reset_first() -> None:
    env = RocketLanderEnv(RocketParams())

    with pytest.raises(RuntimeError, match="must be reset"):
        env.step(np.array([0.0, 0.0], dtype=np.float32))


def test_env_can_truncate_on_step_limit() -> None:
    env = RocketLanderEnv(
        RocketParams(),
        env_config=RLEnvConfig(
            max_episode_steps=1,
            truncation=TruncationConfig(max_abs_dx=10_000.0, max_altitude=10_000.0),
            reset_distribution=InitialStateDistribution(
                dx=UniformRange(0.0, 0.0),
                z=UniformRange(100.0, 100.0),
                vx=UniformRange(0.0, 0.0),
                vz=UniformRange(0.0, 0.0),
                theta=UniformRange(0.0, 0.0),
                omega=UniformRange(0.0, 0.0),
                fuel_ratio=UniformRange(1.0, 1.0),
            ),
        ),
    )
    env.reset(seed=3)

    _observation, _reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.0], dtype=np.float32)
    )

    assert terminated is False
    assert truncated is True
    assert info["episode"]["steps"] == 1


def test_env_reset_accepts_explicit_initial_state() -> None:
    env = RocketLanderEnv(RocketParams(target_x=20.0))
    explicit_state = State(
        x=24.0,
        z=90.0,
        vx=1.0,
        vz=-7.0,
        theta=0.05,
        omega=0.01,
        fuel=4_000.0,
    )

    observation, info = env.reset(options={"initial_state": explicit_state})

    assert observation[0] == pytest.approx(4.0)
    assert info["dx"] == pytest.approx(4.0)


def test_env_can_apply_curriculum_stage() -> None:
    env = RocketLanderEnv(RocketParams())
    stage = CurriculumStage(
        name="test_stage",
        reset_distribution=InitialStateDistribution(
            dx=UniformRange(5.0, 5.0),
            z=UniformRange(100.0, 100.0),
            vx=UniformRange(0.0, 0.0),
            vz=UniformRange(0.0, 0.0),
            theta=UniformRange(0.0, 0.0),
            omega=UniformRange(0.0, 0.0),
            fuel_ratio=UniformRange(1.0, 1.0),
        ),
        truncation=TruncationConfig(max_abs_dx=50.0, max_altitude=200.0),
        promotion=PromotionCriteria(min_success_rate=1.0),
    )

    env.apply_curriculum_stage(stage)
    observation, info = env.reset(seed=1)

    assert env.current_stage_name == "test_stage"
    assert observation[0] == pytest.approx(5.0)
    assert info["dx"] == pytest.approx(5.0)
