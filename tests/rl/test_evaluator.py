import numpy as np
import pytest

from rocket_landing.application.rl.config import EvaluationConfig, RLEnvConfig
from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.application.rl.evaluation.evaluator import evaluate_policy
from rocket_landing.application.rl.reset.distributions import InitialStateDistribution, UniformRange
from rocket_landing.application.rl.termination import TruncationConfig
from rocket_landing.domain.models.params import RocketParams


def test_evaluate_policy_aggregates_episode_metrics() -> None:
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

    report = evaluate_policy(
        env,
        lambda observation: np.array([0.0, 0.0], dtype=np.float32),
        config=EvaluationConfig(episodes=3, deterministic=True, seed=11),
    )

    assert report.metrics.episodes == 3
    assert report.metrics.success_rate == pytest.approx(0.0)
    assert report.metrics.truncation_rate == pytest.approx(1.0)
    assert len(report.episodes) == 3
