"""Resolve Gymnasium termination flags from simulator results."""

from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.application.rl.termination.truncation_rules import (
    TruncationConfig,
    is_truncated,
)
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult


@dataclass(frozen=True, slots=True)
class StepFlags:
    """Final Gymnasium-style terminal flags for one step."""

    terminated: bool
    truncated: bool


def resolve_step_flags(
    *,
    result: StepResult,
    step_count: int,
    params: RocketParams,
    config: TruncationConfig,
    max_episode_steps: int,
) -> StepFlags:
    """Translate domain-level outcome flags into RL episode flags."""

    if result.terminated:
        return StepFlags(terminated=True, truncated=False)
    truncated = is_truncated(
        state=result.state,
        step_count=step_count,
        params=params,
        config=config,
        max_episode_steps=max_episode_steps,
    )
    return StepFlags(terminated=False, truncated=truncated)
