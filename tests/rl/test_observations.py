import math

import pytest

from rocket_landing.application.rl.encoding.observations import build_observation, fuel_ratio_for
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


def test_build_observation_uses_relative_target_offset_and_trig_encoding() -> None:
    params = RocketParams(target_x=15.0, initial_fuel=12_000.0)
    state = State(
        x=18.5,
        z=120.0,
        vx=-2.0,
        vz=-11.0,
        theta=0.25,
        omega=-0.1,
        fuel=6_000.0,
    )

    observation = build_observation(state, params)

    assert observation.tolist() == pytest.approx(
        [3.5, 120.0, -2.0, -11.0, math.sin(0.25), math.cos(0.25), -0.1, 0.5]
    )


def test_fuel_ratio_is_clamped_between_zero_and_one() -> None:
    params = RocketParams(initial_fuel=10.0)

    assert fuel_ratio_for(State(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 12.0), params) == pytest.approx(1.0)
    assert fuel_ratio_for(State(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -2.0), params) == pytest.approx(0.0)
