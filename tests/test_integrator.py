import pytest

from rocket_landing.domain.models.results import ForceVector
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.integrators import SemiImplicitEulerIntegrator


def test_semi_implicit_euler_updates_velocity_before_position() -> None:
    state = State(x=0.0, z=10.0, vx=0.0, vz=0.0, theta=0.0, omega=0.0, fuel=100.0)
    integrator = SemiImplicitEulerIntegrator()

    next_state = integrator.integrate(
        state,
        linear_acceleration=ForceVector(x=1.0, z=-2.0),
        angular_acceleration=0.5,
        dt=0.1,
        fuel=99.0,
    )

    assert next_state.vx == pytest.approx(0.1)
    assert next_state.vz == pytest.approx(-0.2)
    assert next_state.x == pytest.approx(0.01)
    assert next_state.z == pytest.approx(9.98)
    assert next_state.omega == pytest.approx(0.05)
    assert next_state.theta == pytest.approx(0.005)
    assert next_state.fuel == 99.0
