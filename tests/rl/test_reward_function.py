import pytest

from rocket_landing.application.rl.config.reward_config import RewardConfig, RewardWeights
from rocket_landing.application.rl.reward.reward_function import compute_reward
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State


def test_compute_reward_applies_landing_bonus() -> None:
    params = RocketParams(target_x=0.0, initial_fuel=10.0)
    result = StepResult(
        state=State(0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0),
        terminated=True,
        landed=True,
        crashed=False,
        impact_state=State(0.2, 12.0, 0.1, -0.8, 0.01, 0.02, 5.0),
    )

    reward = compute_reward(
        action=Action(throttle=0.2, engine_gimbal=0.01, aero_steer=0.0),
        previous_state=State(0.3, 12.5, 0.2, -1.0, 0.01, 0.02, 5.0),
        result=result,
        params=params,
        config=RewardConfig(
            weights=RewardWeights(
                dx=0.0,
                z=0.0,
                dx_progress=0.0,
                z_progress=0.0,
                vx=0.0,
                vz=0.0,
                tilt=0.0,
                omega=0.0,
                throttle=0.0,
                gimbal=0.0,
                step=0.0,
            ),
            landing_bonus=500.0,
            crash_penalty=300.0,
        ),
    )

    assert reward.terminal == pytest.approx(500.0)
    assert reward.total == pytest.approx(500.0)


def test_compute_reward_applies_crash_penalty() -> None:
    params = RocketParams(target_x=0.0)
    result = StepResult(
        state=State(0.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0),
        terminated=True,
        landed=False,
        crashed=True,
        impact_state=State(10.0, 12.0, 3.0, -12.0, 0.2, 0.4, 5.0),
    )

    reward = compute_reward(
        action=Action(throttle=0.0, engine_gimbal=0.0, aero_steer=0.0),
        previous_state=State(12.0, 14.0, 4.0, -14.0, 0.3, 0.4, 5.0),
        result=result,
        params=params,
        config=RewardConfig(
            weights=RewardWeights(
                dx=0.0,
                z=0.0,
                dx_progress=0.0,
                z_progress=0.0,
                vx=0.0,
                vz=0.0,
                tilt=0.0,
                omega=0.0,
                throttle=0.0,
                gimbal=0.0,
                step=0.0,
            ),
            landing_bonus=500.0,
            crash_penalty=300.0,
        ),
    )

    assert reward.terminal == pytest.approx(-300.0)
    assert reward.total == pytest.approx(-300.0)
