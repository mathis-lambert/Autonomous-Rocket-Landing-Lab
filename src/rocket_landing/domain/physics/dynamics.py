"""Physics model turning state and commands into accelerations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import ForceBreakdown, ForceVector
from rocket_landing.domain.models.state import State


@dataclass(frozen=True, slots=True)
class DynamicsUpdate:
    """Intermediate physics quantities produced before integration."""

    action: Action
    forces: ForceBreakdown
    linear_acceleration: ForceVector
    angular_acceleration: float
    remaining_fuel: float


class BoosterDynamicsModel:
    """Encapsulates the booster equations of motion."""

    def __init__(self, params: RocketParams) -> None:
        self._params = params

    @property
    def params(self) -> RocketParams:
        """Expose the immutable parameter set used by the model."""

        return self._params

    def sanitize_action(self, action: Action) -> Action:
        """Clamp commands to the physically supported control envelope."""

        throttle = min(max(action.throttle, 0.0), 1.0)
        gimbal = min(max(action.gimbal, -self._params.max_gimbal), self._params.max_gimbal)
        return Action(throttle=throttle, gimbal=gimbal)

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

        force_angle = state.theta + action.gimbal
        return ForceVector(
            x=thrust * math.sin(force_angle),
            z=thrust * math.cos(force_angle),
        )

    def gravity_force_for(self, mass: float) -> ForceVector:
        """Return the constant downward gravity force for a given mass."""

        return ForceVector(x=0.0, z=-mass * self._params.gravity)

    def atmospheric_density_for(self, altitude: float) -> float:
        """Return a simple exponential atmosphere density model."""

        clamped_altitude = max(0.0, altitude)
        scale_height = max(self._params.atmosphere_scale_height, 1e-6)
        return self._params.air_density_sea_level * math.exp(-clamped_altitude / scale_height)

    def body_axis_for(self, theta: float) -> tuple[float, float]:
        """Return the world-space longitudinal axis of the booster."""

        return (math.sin(theta), math.cos(theta))

    def body_normal_for(self, theta: float) -> tuple[float, float]:
        """Return the world-space rightward normal axis of the booster."""

        return (math.cos(theta), -math.sin(theta))

    def relative_air_velocity_for(self, state: State) -> ForceVector:
        """Return the vehicle velocity relative to the surrounding air."""

        return ForceVector(x=state.vx, z=state.vz)

    def body_velocity_components_for(self, state: State) -> tuple[float, float]:
        """Project the air-relative velocity onto body longitudinal and normal axes."""

        relative_velocity = self.relative_air_velocity_for(state)
        axis_x, axis_z = self.body_axis_for(state.theta)
        normal_x, normal_z = self.body_normal_for(state.theta)
        axial_speed = (relative_velocity.x * axis_x) + (relative_velocity.z * axis_z)
        lateral_speed = (relative_velocity.x * normal_x) + (relative_velocity.z * normal_z)
        return axial_speed, lateral_speed

    def aerodynamic_force_for(self, state: State) -> ForceVector:
        """Return a 2D aerodynamic force using body-axis axial and lateral components."""

        relative_velocity = self.relative_air_velocity_for(state)
        speed = math.hypot(relative_velocity.x, relative_velocity.z)
        if speed <= 0.0:
            return ForceVector(x=0.0, z=0.0)

        density = self.atmospheric_density_for(state.z)
        axis_x, axis_z = self.body_axis_for(state.theta)
        normal_x, normal_z = self.body_normal_for(state.theta)
        axial_speed, lateral_speed = self.body_velocity_components_for(state)

        axial_force_scalar = (
            -0.5
            * density
            * self._params.axial_drag_coefficient
            * self._params.frontal_area
            * axial_speed
            * abs(axial_speed)
        )
        lateral_force_scalar = (
            -0.5
            * density
            * self._params.side_drag_coefficient
            * self._params.lateral_area
            * lateral_speed
            * abs(lateral_speed)
        )
        return ForceVector(
            x=(axial_force_scalar * axis_x) + (lateral_force_scalar * normal_x),
            z=(axial_force_scalar * axis_z) + (lateral_force_scalar * normal_z),
        )

    def velocity_angle_for(self, state: State) -> float:
        """Return the velocity direction angle in the same convention as theta."""

        relative_velocity = self.relative_air_velocity_for(state)
        if math.hypot(relative_velocity.x, relative_velocity.z) <= 0.0:
            return state.theta
        return math.atan2(relative_velocity.x, relative_velocity.z)

    def angle_of_attack_for(self, state: State) -> float:
        """Return the vehicle angle relative to its current velocity vector."""

        return state.theta - self.velocity_angle_for(state)

    def forces_for(self, state: State, action: Action) -> ForceBreakdown:
        """Return the named force components applied during the current step."""

        safe_action = self.sanitize_action(action)
        thrust = self.thrust_for(state, safe_action)
        engine_force = self.engine_force_for(state, safe_action, thrust)
        gravity_force = self.gravity_force_for(self.current_mass(state))
        aerodynamic_force = self.aerodynamic_force_for(state)
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
        return lever_arm * thrust * math.sin(action.gimbal)

    def aerodynamic_torque_for(self, state: State) -> float:
        """Return aerodynamic restoring and damping torque about the center of mass."""

        relative_velocity = self.relative_air_velocity_for(state)
        speed = math.hypot(relative_velocity.x, relative_velocity.z)
        if speed <= 0.0:
            return 0.0

        density = self.atmospheric_density_for(state.z)
        _, lateral_speed = self.body_velocity_components_for(state)
        lateral_force_scalar = (
            -0.5
            * density
            * self._params.side_drag_coefficient
            * self._params.lateral_area
            * lateral_speed
            * abs(lateral_speed)
        )
        restoring_torque = -self._params.center_of_pressure_offset * lateral_force_scalar
        damping_torque = (
            -0.5
            * density
            * self._params.lateral_area
            * self._params.length
            * self._params.length
            * self._params.angular_damping_coefficient
            * speed
            * state.omega
        )
        return restoring_torque + damping_torque

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
        torque = self.engine_torque_for(safe_action, thrust) + self.aerodynamic_torque_for(state)
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
