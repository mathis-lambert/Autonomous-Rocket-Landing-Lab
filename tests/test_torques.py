from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.physics.dynamics import BoosterDynamicsModel


def test_zero_gimbal_produces_zero_torque() -> None:
    params = RocketParams()
    model = BoosterDynamicsModel(params)

    torque = model.engine_torque_for(
        Action(throttle=0.8, engine_gimbal=0.0, aero_steer=0.0),
        100_000.0,
    )

    assert torque == 0.0


def test_positive_gimbal_produces_positive_torque() -> None:
    params = RocketParams()
    model = BoosterDynamicsModel(params)

    torque = model.engine_torque_for(
        Action(throttle=0.8, engine_gimbal=0.05, aero_steer=0.0),
        100_000.0,
    )

    assert torque > 0.0


def test_moment_of_inertia_is_positive() -> None:
    params = RocketParams()
    model = BoosterDynamicsModel(params)

    inertia = model.moment_of_inertia_for(10_000.0)

    assert inertia > 0.0
