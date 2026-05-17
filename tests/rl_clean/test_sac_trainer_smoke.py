import json
from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.rl import (
    EnvConfig,
    EvaluationConfig,
    RewardConfig,
    RocketLanderEnv,
    TrainingConfig,
    train_sac,
)


def test_train_sac_smoke(tmp_path: Path) -> None:
    params = RocketParams()

    def env_factory() -> RocketLanderEnv:
        return RocketLanderEnv(params, env_config=EnvConfig(), reward_config=RewardConfig())

    summary = train_sac(
        env_factory=env_factory,
        output_dir=tmp_path,
        run_name="smoke",
        config=TrainingConfig(
            total_timesteps=32,
            segment_timesteps=16,
            n_envs=1,
            seed=0,
            progress_bar=False,
            evaluation=EvaluationConfig(episodes=2, seed=0),
        ),
        curriculum=None,
    )

    assert summary.total_timesteps == 32
    assert (tmp_path / "smoke" / "models" / "final_model.zip").exists()
    assert (tmp_path / "smoke" / "evaluations" / "eval_step_000000032.json").exists()


def test_train_sac_can_resume_from_checkpoint(tmp_path: Path) -> None:
    params = RocketParams()

    def env_factory() -> RocketLanderEnv:
        return RocketLanderEnv(params, env_config=EnvConfig(), reward_config=RewardConfig())

    initial = train_sac(
        env_factory=env_factory,
        output_dir=tmp_path,
        run_name="initial",
        config=TrainingConfig(
            total_timesteps=32,
            segment_timesteps=16,
            n_envs=1,
            seed=0,
            progress_bar=False,
            evaluation=EvaluationConfig(episodes=2, seed=0),
        ),
        curriculum=None,
    )
    resume_checkpoint = initial.run_dir / "models" / "final_model.zip"

    resumed = train_sac(
        env_factory=env_factory,
        output_dir=tmp_path,
        run_name="resumed",
        config=TrainingConfig(
            total_timesteps=16,
            segment_timesteps=16,
            n_envs=1,
            seed=1,
            progress_bar=False,
            evaluation=EvaluationConfig(episodes=2, seed=1),
        ),
        curriculum=None,
        initial_model_path=resume_checkpoint,
    )

    manifest = json.loads((tmp_path / "resumed" / "manifest.json").read_text())
    assert resumed.total_timesteps == 16
    assert manifest["initial_model_path"] == resume_checkpoint.as_posix()
    assert (tmp_path / "resumed" / "models" / "final_model.zip").exists()
