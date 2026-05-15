"""Structured reward outputs used for logging and debugging."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Store the individual terms contributing to the step reward."""

    dx: float
    z: float
    dx_progress: float
    z_progress: float
    vx: float
    vz: float
    tilt: float
    omega: float
    throttle: float
    gimbal: float
    step: float
    terminal: float

    @property
    def total(self) -> float:
        """Return the aggregated scalar reward."""

        return (
            self.dx
            + self.z
            + self.dx_progress
            + self.z_progress
            + self.vx
            + self.vz
            + self.tilt
            + self.omega
            + self.throttle
            + self.gimbal
            + self.step
            + self.terminal
        )

    def as_dict(self) -> dict[str, float]:
        """Return a plain mapping suitable for Gymnasium ``info`` payloads."""

        values = asdict(self)
        values["total"] = self.total
        return values
