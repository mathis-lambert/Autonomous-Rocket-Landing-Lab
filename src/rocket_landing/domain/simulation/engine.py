"""Orchestration layer combining dynamics, integration and contact handling."""

from __future__ import annotations

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.collision import GroundContactResolver
from rocket_landing.domain.physics.dynamics import BoosterDynamicsModel
from rocket_landing.domain.physics.integrators import SemiImplicitEulerIntegrator


class SimulationEngine:
    """Coordinates dynamics, numerical integration, and contact resolution."""

    def __init__(
        self,
        params: RocketParams,
        *,
        dynamics: BoosterDynamicsModel | None = None,
        integrator: SemiImplicitEulerIntegrator | None = None,
        collision_resolver: GroundContactResolver | None = None,
    ) -> None:
        self._params = params
        self.dynamics = dynamics or BoosterDynamicsModel(params)
        self._integrator = integrator or SemiImplicitEulerIntegrator()
        self._collision_resolver = collision_resolver or GroundContactResolver(params)

    def step(self, state: State, action: Action, dt: float) -> StepResult:
        """Advance the simulation by one step and resolve terminal events."""

        update = self.dynamics.evaluate(state, action, dt)
        next_state = self._integrator.integrate(
            state,
            linear_acceleration=update.linear_acceleration,
            angular_acceleration=update.angular_acceleration,
            dt=dt,
            fuel=update.remaining_fuel,
        )
        return self._collision_resolver.resolve(next_state)
