"""Physics model turning state and commands into accelerations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import ForceBreakdown, ForceVector
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.aerodynamics import AerodynamicLoads, AerodynamicsModel


@dataclass(frozen=True, slots=True)
class DynamicsUpdate:
    """Intermediate physics quantities produced before integration."""

    action: Action
    forces: ForceBreakdown
    linear_acceleration: ForceVector
    angular_acceleration: float
    remaining_fuel: float


class BoosterDynamicsModel:
    """Encapsulates propulsion, aerodynamics and rigid-body dynamics."""

    def __init__(self, params: RocketParams) -> None:
        self._params = params
        self._aerodynamics = AerodynamicsModel(params)

    @property
    def params(self) -> RocketParams:
        """Expose the immutable parameter set used by the model."""

        return self._params

    def sanitize_action(self, action: Action) -> Action:
        """Clamp commands to the physically supported control envelope."""

        throttle = min(max(action.throttle, 0.0), 1.0)
        engine_gimbal = min(
            max(action.engine_gimbal, -self._params.max_gimbal),
            self._params.max_gimbal,
        )
        aero_steer = min(max(action.aero_steer, -1.0), 1.0)
        return Action(
            throttle=throttle,
            engine_gimbal=engine_gimbal,
            aero_steer=aero_steer,
        )

    def current_mass(self, state: State) -> float:
        """Return dry mass plus the current propellant mass."""

        return self._params.dry_mass + state.fuel

    def thrust_for(self, state: State, action: Action) -> float:
        """Convert a throttle command into physical thrust in newtons."""

        if state.fuel <= 0.0:
            return 0.0
        return action.throttle * self._params.max_thrust

    def engine_force_for(self, state: State, action: Action, thrust: float) -> ForceVector:
        """Resolve engine thrust into world-space horizontal and vertical components."""

        force_angle = state.theta + action.engine_gimbal
        return ForceVector(
            x=thrust * math.sin(force_angle),
            z=thrust * math.cos(force_angle),
        )

    def gravity_force_for(self, mass: float) -> ForceVector:
        """Return the constant downward gravity force for a given mass."""

        return ForceVector(x=0.0, z=-mass * self._params.gravity)

    def atmospheric_density_for(self, altitude: float) -> float:
        """Return the atmosphere density at the provided altitude."""

        return self._aerodynamics.atmospheric_density_for(altitude)

    def body_axis_for(self, theta: float) -> tuple[float, float]:
        """Return the world-space longitudinal axis of the booster."""

        return self._aerodynamics.body_axis_for(theta)

    def body_normal_for(self, theta: float) -> tuple[float, float]:
        """Return the world-space rightward normal axis of the booster."""

        return self._aerodynamics.body_normal_for(theta)

    def relative_air_velocity_for(self, state: State) -> ForceVector:
        """Return the vehicle velocity relative to the surrounding air."""

        return self._aerodynamics.relative_air_velocity_for(state)

    def body_velocity_components_for(self, state: State) -> tuple[float, float]:
        """Project the air-relative velocity onto body longitudinal and normal axes."""

        return self._aerodynamics.body_velocity_components_for(state)

    def velocity_angle_for(self, state: State) -> float:
        """Return the air-relative velocity direction in the same convention as theta."""

        return self._aerodynamics.velocity_angle_for(state)

    def angle_of_attack_for(self, state: State) -> float:
        """Return the vehicle angle relative to the air-relative velocity vector."""

        return self._aerodynamics.angle_of_attack_for(state)

    def aerodynamic_loads_for(self, state: State, action: Action) -> AerodynamicLoads:
        """Resolve aerodynamic forces and moments for the current state and steering."""

        safe_action = self.sanitize_action(action)
        return self._aerodynamics.evaluate(state, safe_action.aero_steer)

    def aerodynamic_force_for(self, state: State, action: Action) -> ForceVector:
        """Return the total aerodynamic force for the current state and action."""

        return self.aerodynamic_loads_for(state, action).total_force

    def forces_for(self, state: State, action: Action) -> ForceBreakdown:
        """Return the named force components applied during the current step."""

        safe_action = self.sanitize_action(action)
        thrust = self.thrust_for(state, safe_action)
        engine_force = self.engine_force_for(state, safe_action, thrust)
        gravity_force = self.gravity_force_for(self.current_mass(state))
        aerodynamic_force = self.aerodynamic_force_for(state, safe_action)
        total_force = ForceVector(
            x=engine_force.x + gravity_force.x + aerodynamic_force.x,
            z=engine_force.z + gravity_force.z + aerodynamic_force.z,
        )
        return ForceBreakdown(
            engine=engine_force,
            gravity=gravity_force,
            aerodynamic=aerodynamic_force,
            total=total_force,
        )

    def total_force_for(self, state: State, action: Action) -> ForceVector:
        """Return the net force applied during the current step."""

        return self.forces_for(state, action).total

    def moment_of_inertia_for(self, mass: float) -> float:
        """Approximate the booster as a slender rod about its center of mass."""

        return (mass * self._params.length * self._params.length) / 12.0

    def engine_torque_for(self, action: Action, thrust: float) -> float:
        """Return the torque produced by a gimbaled engine."""

        lever_arm = self._params.length / 2.0
        return lever_arm * thrust * math.sin(action.engine_gimbal)

    def remaining_fuel_for(self, state: State, action: Action, dt: float) -> float:
        """Integrate propellant usage over one simulation step."""

        fuel_burn = action.throttle * self._params.fuel_flow_rate * dt
        return max(0.0, state.fuel - fuel_burn)

    def evaluate(self, state: State, action: Action, dt: float) -> DynamicsUpdate:
        """Evaluate one step of continuous dynamics without mutating the state."""

        safe_action = self.sanitize_action(action)
        mass = self.current_mass(state)
        thrust = self.thrust_for(state, safe_action)
        forces = self.forces_for(state, safe_action)
        aerodynamic_loads = self.aerodynamic_loads_for(state, safe_action)
        torque = self.engine_torque_for(safe_action, thrust) + aerodynamic_loads.total_torque
        inertia = self.moment_of_inertia_for(mass)
        return DynamicsUpdate(
            action=safe_action,
            forces=forces,
            linear_acceleration=ForceVector(
                x=forces.total.x / mass,
                z=forces.total.z / mass,
            ),
            angular_acceleration=torque / inertia,
            remaining_fuel=self.remaining_fuel_for(state, safe_action, dt),
        )
