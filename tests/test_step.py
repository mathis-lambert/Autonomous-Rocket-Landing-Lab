from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.simulation.engine import SimulationEngine


def test_no_thrust_causes_descent() -> None:
    params = RocketParams()
    engine = SimulationEngine(params)
    state = State(x=0.0, z=100.0, vx=0.0, vz=0.0, theta=0.0, omega=0.0, fuel=100.0)

    result = engine.step(state, Action(throttle=0.0, gimbal=0.0), dt=0.1)

    assert result.state.vz < 0.0
    assert result.state.z < state.z
    assert result.terminated is False


def test_high_throttle_slows_descent() -> None:
    params = RocketParams()
    engine = SimulationEngine(params)
    state = State(x=0.0, z=100.0, vx=0.0, vz=-10.0, theta=0.0, omega=0.0, fuel=100.0)

    result = engine.step(state, Action(throttle=1.0, gimbal=0.0), dt=0.1)

    assert result.state.vz > state.vz


def test_gimbaled_thrust_changes_attitude() -> None:
    params = RocketParams()
    engine = SimulationEngine(params)
    state = State(x=0.0, z=100.0, vx=0.0, vz=-5.0, theta=0.0, omega=0.0, fuel=100.0)

    result = engine.step(state, Action(throttle=0.8, gimbal=0.05), dt=0.1)

    assert result.state.omega > 0.0
    assert result.state.theta > 0.0


def test_fuel_decreases_when_throttle_is_applied() -> None:
    params = RocketParams()
    engine = SimulationEngine(params)
    state = State(x=0.0, z=100.0, vx=0.0, vz=-5.0, theta=0.0, omega=0.0, fuel=100.0)

    result = engine.step(state, Action(throttle=0.5, gimbal=0.0), dt=0.5)

    assert result.state.fuel < state.fuel
