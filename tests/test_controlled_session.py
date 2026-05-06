from rocket_landing.application.use_cases.run_controlled_session import ControlledSimulationSession
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams


def test_controlled_session_reset_creates_initial_history_point() -> None:
    session = ControlledSimulationSession(params=RocketParams(), dt=0.02)

    assert len(session.history.states) == 1
    assert session.step_count == 0
    assert session.is_finished is False


def test_controlled_session_step_appends_history() -> None:
    session = ControlledSimulationSession(params=RocketParams(), dt=0.02)

    result = session.step(Action(throttle=0.0, gimbal=0.0))

    assert len(session.history.states) == 2
    assert session.step_count == 1
    assert result.state.z < session.history.states[0].z
