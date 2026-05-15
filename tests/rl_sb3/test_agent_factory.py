from pathlib import Path

import pytest

pytest.importorskip("stable_baselines3")

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.infrastructure.rl.sb3 import SB3TrainerConfig, build_sac_model


def test_build_sac_model_creates_expected_algo(tmp_path: Path) -> None:
    env = VecMonitor(DummyVecEnv([lambda: RocketLanderEnv(RocketParams())]))

    model = build_sac_model(
        env,
        config=SB3TrainerConfig(),
        tensorboard_log_dir=tmp_path,
    )

    assert model.__class__.__name__ == "SAC"
    env.close()
