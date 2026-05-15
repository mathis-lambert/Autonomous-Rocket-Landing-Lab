from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from rocket_landing.application.rl.config import EvaluationConfig, RewardConfig, RLEnvConfig
from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.application.rl.reset.distributions import (
    InitialStateDistribution,
    UniformRange,
)
from rocket_landing.application.rl.termination import TruncationConfig
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.infrastructure.rl.sb3 import (
    SACHyperparameters,
    SB3Trainer,
    SB3TrainerConfig,
)


def test_trainer_runs_short_smoke_training(tmp_path: Path) -> None:
    params = RocketParams()
    env_config = RLEnvConfig(
        max_episode_steps=8,
        truncation=TruncationConfig(max_abs_dx=10_000.0, max_altitude=10_000.0),
        reset_distribution=InitialStateDistribution(
            dx=UniformRange(0.0, 0.0),
            z=UniformRange(50.0, 50.0),
            vx=UniformRange(0.0, 0.0),
            vz=UniformRange(-1.0, -1.0),
            theta=UniformRange(0.0, 0.0),
            omega=UniformRange(0.0, 0.0),
            fuel_ratio=UniformRange(1.0, 1.0),
        ),
    )

    def env_factory() -> RocketLanderEnv:
        return RocketLanderEnv(params, env_config=env_config, reward_config=RewardConfig())

    trainer = SB3Trainer(
        SB3TrainerConfig(
            total_timesteps=32,
            train_segment_timesteps=16,
            n_envs=1,
            progress_bar=False,
            evaluation=EvaluationConfig(episodes=2),
            sac=SACHyperparameters(
                learning_rate=3e-4,
                buffer_size=128,
                learning_starts=0,
                batch_size=8,
                train_freq=1,
                gradient_steps=1,
            ),
        )
    )

    summary = trainer.train(
        env_factory=env_factory,
        output_dir=tmp_path,
        run_name="smoke",
    )

    assert summary.total_timesteps == 32
    assert (tmp_path / "smoke" / "models" / "final_model.zip").exists()
