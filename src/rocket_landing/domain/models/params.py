"""Immutable parameter sets for physics and landing constraints.

These values define both the mechanical envelope of the booster and the
thresholds used to classify a touchdown as safe or unsafe.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RocketParams:
    """Physical parameters and landing thresholds for the booster.

    All units are SI unless otherwise noted.
    """

    gravity: float = 9.81
    dry_mass: float = 22_000.0
    initial_fuel: float = 12_000.0
    max_thrust: float = 900_000.0
    fuel_flow_rate: float = 180.0
    length: float = 24.0
    radius: float = 1.8
    max_gimbal: float = 0.15
    max_landing_vz: float = 3.0
    max_landing_vx: float = 1.5
    max_landing_theta: float = 0.10
    max_landing_omega: float = 0.25

    @property
    def initial_mass(self) -> float:
        """Return the starting mass before any propellant is burned."""

        return self.dry_mass + self.initial_fuel

    @property
    def frontal_area(self) -> float:
        """Return the circular frontal reference area in square meters."""

        return 3.141592653589793 * self.radius * self.radius
