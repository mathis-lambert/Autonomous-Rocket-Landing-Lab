from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.domain.models.state import State


@dataclass(frozen=True, slots=True)
class ForceVector:
    x: float
    z: float


@dataclass(frozen=True, slots=True)
class StepResult:
    state: State
    terminated: bool
    landed: bool
    crashed: bool
