from __future__ import annotations

from rocket_landing.domain.models.results import ForceVector
from rocket_landing.domain.models.state import State


class SemiImplicitEulerIntegrator:
    """Numerical integrator responsible only for state advancement."""

    def integrate(
        self,
        state: State,
        *,
        linear_acceleration: ForceVector,
        angular_acceleration: float,
        dt: float,
        fuel: float,
    ) -> State:
        vx = state.vx + linear_acceleration.x * dt
        vz = state.vz + linear_acceleration.z * dt
        x = state.x + vx * dt
        z = state.z + vz * dt

        omega = state.omega + angular_acceleration * dt
        theta = state.theta + omega * dt

        return State(
            x=x,
            z=z,
            vx=vx,
            vz=vz,
            theta=theta,
            omega=omega,
            fuel=fuel,
        )
