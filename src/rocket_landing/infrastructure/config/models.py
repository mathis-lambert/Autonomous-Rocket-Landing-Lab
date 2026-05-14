"""Configuration models resolved from YAML scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


@dataclass(frozen=True, slots=True)
class ManualControlConfig:
    """Configurable rates used by the live keyboard controller."""

    throttle_rate: float = 1.15
    engine_gimbal_rate: float = 1.25
    aero_steer_rate: float = 4.0
    steering_return_rate: float = 1.65


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Fully resolved live simulation configuration."""

    name: str
    params: RocketParams
    initial_state: State
    controls: ManualControlConfig
    description: str | None = None
