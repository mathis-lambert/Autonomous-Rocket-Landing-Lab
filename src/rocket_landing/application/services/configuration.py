"""Configuration models used to describe simulation scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Fully resolved simulation configuration.

    A configuration represents one runnable scenario: vehicle and environment
    parameters, landing constraints, and the initial vehicle state.
    """

    name: str
    params: RocketParams
    initial_state: State
    description: str | None = None
