from __future__ import annotations

import math
from dataclasses import dataclass

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import ForceVector
from rocket_landing.domain.models.state import State


@dataclass(frozen=True, slots=True)
class DynamicsUpdate:
    action: Action
    linear_acceleration: ForceVector
    angular_acceleration: float
    remaining_fuel: float


class BoosterDynamicsModel:
    """Encapsulates the booster equations of motion."""

    def __init__(self, params: RocketParams) -> None:
        self._params = params

    @property
    def params(self) -> RocketParams:
        return self._params

    def sanitize_action(self, action: Action) -> Action:
        throttle = min(max(action.throttle, 0.0), 1.0)
        gimbal = min(max(action.gimbal, -self._params.max_gimbal), self._params.max_gimbal)
        return Action(throttle=throttle, gimbal=gimbal)

    def current_mass(self, state: State) -> float:
        return self._params.dry_mass + state.fuel

    def thrust_for(self, state: State, action: Action) -> float:
        if state.fuel <= 0.0:
            return 0.0
        return action.throttle * self._params.max_thrust

    def engine_force_for(self, state: State, action: Action, thrust: float) -> ForceVector:
        force_angle = state.theta + action.gimbal
        return ForceVector(
            x=thrust * math.sin(force_angle),
            z=thrust * math.cos(force_angle),
        )

    def gravity_force_for(self, mass: float) -> ForceVector:
        return ForceVector(x=0.0, z=-mass * self._params.gravity)

    def total_force_for(self, state: State, action: Action) -> ForceVector:
        safe_action = self.sanitize_action(action)
        thrust = self.thrust_for(state, safe_action)
        engine_force = self.engine_force_for(state, safe_action, thrust)
        gravity_force = self.gravity_force_for(self.current_mass(state))
        return ForceVector(
            x=engine_force.x + gravity_force.x,
            z=engine_force.z + gravity_force.z,
        )

    def moment_of_inertia_for(self, mass: float) -> float:
        return (mass * self._params.length * self._params.length) / 12.0

    def engine_torque_for(self, action: Action, thrust: float) -> float:
        lever_arm = self._params.length / 2.0
        return lever_arm * thrust * math.sin(action.gimbal)

    def remaining_fuel_for(self, state: State, action: Action, dt: float) -> float:
        fuel_burn = action.throttle * self._params.fuel_flow_rate * dt
        return max(0.0, state.fuel - fuel_burn)

    def evaluate(self, state: State, action: Action, dt: float) -> DynamicsUpdate:
        safe_action = self.sanitize_action(action)
        mass = self.current_mass(state)
        thrust = self.thrust_for(state, safe_action)
        total_force = self.total_force_for(state, safe_action)
        torque = self.engine_torque_for(safe_action, thrust)
        inertia = self.moment_of_inertia_for(mass)
        return DynamicsUpdate(
            action=safe_action,
            linear_acceleration=ForceVector(
                x=total_force.x / mass,
                z=total_force.z / mass,
            ),
            angular_acceleration=torque / inertia,
            remaining_fuel=self.remaining_fuel_for(state, safe_action, dt),
        )
