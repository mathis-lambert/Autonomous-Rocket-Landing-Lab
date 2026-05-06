from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class State:
    """Continuous 2D rigid-body state for the booster."""

    x: float
    z: float
    vx: float
    vz: float
    theta: float
    omega: float
    fuel: float

    @property
    def speed(self) -> float:
        return (self.vx * self.vx + self.vz * self.vz) ** 0.5
