"""Aerodynamic helper model for the booster physics."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import ForceVector
from rocket_landing.domain.models.state import State


@dataclass(frozen=True, slots=True)
class AerodynamicLoads:
    """Resolved aerodynamic loads for one vehicle state and steering command."""

    density: float
    relative_velocity: ForceVector
    angle_of_attack: float
    axial_speed: float
    lateral_speed: float
    passive_force: ForceVector
    control_force: ForceVector
    total_force: ForceVector
    passive_torque: float
    control_torque: float
    damping_torque: float

    @property
    def total_torque(self) -> float:
        """Return the sum of passive, controlled and damping moments."""

        return self.passive_torque + self.control_torque + self.damping_torque


class AerodynamicsModel:
    """Evaluate aerodynamic forces and moments from state and steering inputs."""

    def __init__(self, params: RocketParams) -> None:
        self._params = params

    def atmospheric_density_for(self, altitude: float) -> float:
        """Return a simple exponential atmosphere density model."""

        clamped_altitude = max(0.0, altitude)
        scale_height = max(self._params.atmosphere_scale_height, 1e-6)
        return self._params.air_density_sea_level * math.exp(-clamped_altitude / scale_height)

    @staticmethod
    def body_axis_for(theta: float) -> tuple[float, float]:
        """Return the world-space longitudinal axis of the booster."""

        return (math.sin(theta), math.cos(theta))

    @staticmethod
    def body_normal_for(theta: float) -> tuple[float, float]:
        """Return the world-space rightward normal axis of the booster."""

        return (math.cos(theta), -math.sin(theta))

    @staticmethod
    def relative_air_velocity_for(state: State) -> ForceVector:
        """Return the vehicle velocity relative to the surrounding air."""

        return ForceVector(x=state.vx, z=state.vz)

    def body_velocity_components_for(self, state: State) -> tuple[float, float]:
        """Project air-relative velocity onto body longitudinal and normal axes."""

        relative_velocity = self.relative_air_velocity_for(state)
        axis_x, axis_z = self.body_axis_for(state.theta)
        normal_x, normal_z = self.body_normal_for(state.theta)
        axial_speed = (relative_velocity.x * axis_x) + (relative_velocity.z * axis_z)
        lateral_speed = (relative_velocity.x * normal_x) + (relative_velocity.z * normal_z)
        return axial_speed, lateral_speed

    def velocity_angle_for(self, state: State) -> float:
        """Return the air-relative velocity direction in the same convention as theta."""

        relative_velocity = self.relative_air_velocity_for(state)
        if math.hypot(relative_velocity.x, relative_velocity.z) <= 0.0:
            return state.theta
        return math.atan2(relative_velocity.x, relative_velocity.z)

    def angle_of_attack_for(self, state: State) -> float:
        """Return the vehicle angle relative to the air-relative velocity vector."""

        return state.theta - self.velocity_angle_for(state)

    def evaluate(self, state: State, aero_steer: float) -> AerodynamicLoads:
        """Resolve aerodynamic forces and moments for the given state."""

        relative_velocity = self.relative_air_velocity_for(state)
        speed = math.hypot(relative_velocity.x, relative_velocity.z)
        density = self.atmospheric_density_for(state.z)
        angle_of_attack = self.angle_of_attack_for(state)

        if speed <= 0.0:
            zero = ForceVector(x=0.0, z=0.0)
            return AerodynamicLoads(
                density=density,
                relative_velocity=relative_velocity,
                angle_of_attack=angle_of_attack,
                axial_speed=0.0,
                lateral_speed=0.0,
                passive_force=zero,
                control_force=zero,
                total_force=zero,
                passive_torque=0.0,
                control_torque=0.0,
                damping_torque=0.0,
            )

        dynamic_pressure = 0.5 * density * speed * speed
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
        passive_normal_force_scalar = (
            -0.5
            * density
            * self._params.side_drag_coefficient
            * self._params.lateral_area
            * lateral_speed
            * abs(lateral_speed)
        )
        control_normal_force_scalar = (
            -dynamic_pressure
            * self._params.control_surface_force_coefficient
            * self._params.lateral_area
            * aero_steer
        )

        passive_force = self._resolve_force(
            axis_x,
            axis_z,
            normal_x,
            normal_z,
            axial_force_scalar,
            passive_normal_force_scalar,
        )
        control_force = self._resolve_force(
            axis_x,
            axis_z,
            normal_x,
            normal_z,
            0.0,
            control_normal_force_scalar,
        )
        total_force = ForceVector(
            x=passive_force.x + control_force.x,
            z=passive_force.z + control_force.z,
        )

        passive_torque = -self._params.center_of_pressure_offset * passive_normal_force_scalar
        control_torque = -self._params.center_of_pressure_offset * control_normal_force_scalar
        damping_torque = (
            -dynamic_pressure
            * self._params.lateral_area
            * self._params.length
            * self._params.angular_damping_coefficient
            * state.omega
        )
        return AerodynamicLoads(
            density=density,
            relative_velocity=relative_velocity,
            angle_of_attack=angle_of_attack,
            axial_speed=axial_speed,
            lateral_speed=lateral_speed,
            passive_force=passive_force,
            control_force=control_force,
            total_force=total_force,
            passive_torque=passive_torque,
            control_torque=control_torque,
            damping_torque=damping_torque,
        )

    @staticmethod
    def _resolve_force(
        axis_x: float,
        axis_z: float,
        normal_x: float,
        normal_z: float,
        axial_force_scalar: float,
        normal_force_scalar: float,
    ) -> ForceVector:
        return ForceVector(
            x=(axial_force_scalar * axis_x) + (normal_force_scalar * normal_x),
            z=(axial_force_scalar * axis_z) + (normal_force_scalar * normal_z),
        )
