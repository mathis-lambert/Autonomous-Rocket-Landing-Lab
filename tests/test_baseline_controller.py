from rocket_landing.application.control.baseline import BaselineLandingController
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


def test_baseline_controller_outputs_bounded_action() -> None:
    params = RocketParams()
    controller = BaselineLandingController(params)
    state = State(x=20.0, z=80.0, vx=3.0, vz=-12.0, theta=0.05, omega=0.1, fuel=2_000.0)

    controller.reset(state)
    action = controller.compute_action(state, dt=0.02)

    assert 0.0 <= action.throttle <= 1.0
    assert -params.max_gimbal <= action.gimbal <= params.max_gimbal


def test_baseline_controller_tries_to_tilt_back_toward_pad() -> None:
    params = RocketParams()
    controller = BaselineLandingController(params)
    state = State(x=25.0, z=100.0, vx=0.0, vz=-10.0, theta=0.0, omega=0.0, fuel=2_000.0)

    controller.reset(state)
    action = controller.compute_action(state, dt=0.02)

    assert action.gimbal < 0.0
