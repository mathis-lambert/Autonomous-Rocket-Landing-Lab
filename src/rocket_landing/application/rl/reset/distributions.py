"""Sampling distributions for RL environment resets."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UniformRange:
    """Simple uniform scalar range used by the reset sampler."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("uniform range high bound must be greater than or equal to low")


@dataclass(frozen=True, slots=True)
class InitialStateDistribution:
    """Independent ranges used to sample a reset state."""

    dx: UniformRange
    z: UniformRange
    vx: UniformRange
    vz: UniformRange
    theta: UniformRange
    omega: UniformRange
    fuel_ratio: UniformRange


def default_initial_state_distribution() -> InitialStateDistribution:
    """Return a conservative reset distribution suitable for early training."""

    return InitialStateDistribution(
        dx=UniformRange(-25.0, 25.0),
        z=UniformRange(80.0, 140.0),
        vx=UniformRange(-6.0, 6.0),
        vz=UniformRange(-20.0, -8.0),
        theta=UniformRange(-0.10, 0.10),
        omega=UniformRange(-0.08, 0.08),
        fuel_ratio=UniformRange(0.55, 1.0),
    )
