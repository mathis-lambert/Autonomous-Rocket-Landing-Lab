"""Domain state objects used by the 2D booster simulation.

The project uses a 2D planar convention:
- ``x`` grows to the right
- ``z`` grows upward
- ``theta = 0`` means the booster is perfectly vertical
- positive ``theta`` means the top of the booster leans to the right
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def orientation_error(theta: float) -> float:
    """Return the smallest physically equivalent attitude error in radians."""

    return abs(math.atan2(math.sin(theta), math.cos(theta)))


@dataclass(frozen=True, slots=True)
class State:
    """Continuous 2D rigid-body state for the booster.

    Attributes:
        x: Horizontal position in meters.
        z: Altitude above ground in meters.
        vx: Horizontal velocity in meters per second.
        vz: Vertical velocity in meters per second.
        theta: Attitude angle in radians.
        omega: Angular velocity in radians per second.
        fuel: Remaining propellant mass in kilograms.
    """

    x: float
    z: float
    vx: float
    vz: float
    theta: float
    omega: float
    fuel: float

    def horizontal_error(self, target_x: float) -> float:
        """Return the signed lateral offset from the configured landing target."""

        return self.x - target_x

    @property
    def speed(self) -> float:
        """Return the Euclidean norm of the translational velocity."""

        return (self.vx * self.vx + self.vz * self.vz) ** 0.5
