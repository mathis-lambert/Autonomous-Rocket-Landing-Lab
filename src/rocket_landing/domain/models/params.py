from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RocketParams:
    """Physical parameters and landing thresholds for the booster."""

    gravity: float = 9.81
    dry_mass: float = 22_000.0
    initial_fuel: float = 8_000.0
    max_thrust: float = 360_000.0
    fuel_flow_rate: float = 120.0
    length: float = 24.0
    radius: float = 1.8
    max_gimbal: float = 0.15
    max_landing_vz: float = 3.0
    max_landing_vx: float = 1.5
    max_landing_theta: float = 0.10
    max_landing_omega: float = 0.25

    @property
    def initial_mass(self) -> float:
        return self.dry_mass + self.initial_fuel

    @property
    def frontal_area(self) -> float:
        return 3.141592653589793 * self.radius * self.radius
