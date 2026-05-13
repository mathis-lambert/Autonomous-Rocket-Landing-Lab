from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.simulation.engine import SimulationEngine


def test_drag_reduces_horizontal_speed_without_thrust() -> None:
    params = RocketParams(axial_drag_coefficient=0.4, side_drag_coefficient=1.2)
    engine = SimulationEngine(params)
    state = State(x=0.0, z=800.0, vx=60.0, vz=0.0, theta=0.0, omega=0.0, fuel=100.0)

    result = engine.step(state, Action(throttle=0.0, gimbal=0.0), dt=0.1)

    assert result.state.vx < state.vx


def test_aerodynamic_torque_rotates_vehicle_toward_velocity_vector() -> None:
    params = RocketParams(
        side_drag_coefficient=1.15,
        center_of_pressure_offset=7.0,
        angular_damping_coefficient=0.12,
    )
    engine = SimulationEngine(params)
    state = State(x=0.0, z=1_200.0, vx=10.0, vz=55.0, theta=0.35, omega=0.0, fuel=100.0)

    result = engine.step(state, Action(throttle=0.0, gimbal=0.0), dt=0.2)

    assert result.state.omega < 0.0
    assert result.state.theta < state.theta


def test_side_slip_generates_lateral_aerodynamic_force() -> None:
    params = RocketParams(axial_drag_coefficient=0.3, side_drag_coefficient=1.3)
    engine = SimulationEngine(params)
    state = State(x=0.0, z=900.0, vx=35.0, vz=-5.0, theta=0.0, omega=0.0, fuel=100.0)

    result = engine.dynamics.forces_for(state, Action(throttle=0.0, gimbal=0.0))

    assert result.aerodynamic.x < 0.0
