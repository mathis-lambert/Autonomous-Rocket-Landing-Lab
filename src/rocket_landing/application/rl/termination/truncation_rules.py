"""Non-physical truncation rules for RL episodes."""

from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


@dataclass(frozen=True, slots=True)
class TruncationConfig:
    """Hard training bounds used to stop hopeless episodes early."""

    max_abs_dx: float = 250.0
    max_altitude: float = 500.0


def is_truncated(
    *,
    state: State,
    step_count: int,
    params: RocketParams,
    config: TruncationConfig,
    max_episode_steps: int,
) -> bool:
    """Return whether the rollout should stop for training-only reasons."""

    if step_count >= max_episode_steps:
        return True
    if abs(state.horizontal_error(params.target_x)) > config.max_abs_dx:
        return True
    return state.z > config.max_altitude
