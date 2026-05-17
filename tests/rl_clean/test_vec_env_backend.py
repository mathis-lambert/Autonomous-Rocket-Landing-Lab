import pytest

pytest.importorskip("stable_baselines3")

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.rl import EnvConfig, RewardConfig, RocketLanderEnv
from rocket_landing.rl.sb3 import _make_vec_env


def test_make_vec_env_uses_dummy_for_single_env() -> None:
    params = RocketParams()

    def env_factory() -> RocketLanderEnv:
        return RocketLanderEnv(params, env_config=EnvConfig(), reward_config=RewardConfig())

    vec_env = _make_vec_env(env_factory, 1)
    try:
        assert vec_env.venv.__class__.__name__ == "DummyVecEnv"
    finally:
        vec_env.close()

