from rocket_landing.application.use_cases.run_controlled_session import ControlledSimulationSession
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


def test_controlled_session_reset_creates_initial_history_point() -> None:
    initial_state = State(
        x=12.0,
        z=140.0,
        vx=-3.0,
        vz=-11.0,
        theta=0.04,
        omega=0.02,
        fuel=1_500.0,
    )
    session = ControlledSimulationSession(
        params=RocketParams(),
        dt=0.02,
        initial_state=initial_state,
    )

    assert len(session.history.states) == 1
    assert session.step_count == 0
    assert session.is_finished is False
    assert session.state == initial_state


def test_controlled_session_step_appends_history() -> None:
    session = ControlledSimulationSession(
        params=RocketParams(),
        dt=0.02,
        initial_state=State(
            x=0.0,
            z=120.0,
            vx=0.0,
            vz=-15.0,
            theta=0.0,
            omega=0.0,
            fuel=8_000.0,
        ),
    )

    result = session.step(Action(throttle=0.0, gimbal=0.0))

    assert len(session.history.states) == 2
    assert session.step_count == 1
    assert result.state.z < session.history.states[0].z


def test_controlled_session_reset_restores_configured_initial_state() -> None:
    initial_state = State(
        x=5.0,
        z=90.0,
        vx=2.0,
        vz=-7.0,
        theta=0.01,
        omega=-0.03,
        fuel=4_000.0,
    )
    session = ControlledSimulationSession(
        params=RocketParams(),
        dt=0.02,
        initial_state=initial_state,
    )

    session.step(Action(throttle=0.2, gimbal=0.0))
    session.reset()

    assert session.state == initial_state
    assert session.step_count == 0
    assert len(session.history.states) == 1
