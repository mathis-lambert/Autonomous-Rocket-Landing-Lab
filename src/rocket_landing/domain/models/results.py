"""Result value objects shared across the simulation pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.domain.models.state import State


@dataclass(frozen=True, slots=True)
class ForceVector:
    """Two-dimensional vector used for forces and accelerations."""

    x: float
    z: float


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of one simulation step after contact resolution."""

    state: State
    terminated: bool
    landed: bool
    crashed: bool
