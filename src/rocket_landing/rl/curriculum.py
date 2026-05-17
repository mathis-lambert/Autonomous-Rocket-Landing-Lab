"""Curriculum scheduling and promotion checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rocket_landing.rl.curriculum_stages import build_default_curriculum, find_stage_by_name
from rocket_landing.rl.types import CurriculumStage

__all__ = [
    "CurriculumScheduler",
    "ProgressMetrics",
    "build_default_curriculum",
    "find_stage_by_name",
    "meets_promotion_criteria",
]


class ProgressMetrics(Protocol):
    """Minimal metric contract used for curriculum promotion."""

    episodes: int
    success_rate: float
    crash_rate: float
    truncation_rate: float
    mean_abs_final_dx: float
    mean_abs_final_vx: float
    mean_abs_final_vz: float
    mean_abs_final_theta: float
    mean_abs_final_omega: float


def meets_promotion_criteria(stage: CurriculumStage, metrics: ProgressMetrics) -> bool:
    """Return whether metrics are strong enough to leave a curriculum stage."""

    criteria = stage.promotion
    if metrics.episodes < criteria.min_eval_episodes:
        return False
    if metrics.success_rate < criteria.min_success_rate:
        return False
    if criteria.max_crash_rate is not None and metrics.crash_rate > criteria.max_crash_rate:
        return False
    if (
        criteria.max_truncation_rate is not None
        and metrics.truncation_rate > criteria.max_truncation_rate
    ):
        return False
    if (
        criteria.max_mean_abs_final_dx is not None
        and metrics.mean_abs_final_dx > criteria.max_mean_abs_final_dx
    ):
        return False
    if (
        criteria.max_mean_abs_final_vx is not None
        and metrics.mean_abs_final_vx > criteria.max_mean_abs_final_vx
    ):
        return False
    if (
        criteria.max_mean_abs_final_vz is not None
        and metrics.mean_abs_final_vz > criteria.max_mean_abs_final_vz
    ):
        return False
    if (
        criteria.max_mean_abs_final_theta is not None
        and metrics.mean_abs_final_theta > criteria.max_mean_abs_final_theta
    ):
        return False
    if (
        criteria.max_mean_abs_final_omega is not None
        and metrics.mean_abs_final_omega > criteria.max_mean_abs_final_omega
    ):
        return False
    return True


@dataclass(slots=True)
class CurriculumScheduler:
    """Stateful curriculum scheduler."""

    stages: tuple[CurriculumStage, ...]
    _current_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("curriculum scheduler requires at least one stage")

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[self._current_index]

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def is_final_stage(self) -> bool:
        return self._current_index >= len(self.stages) - 1

    @property
    def next_stage(self) -> CurriculumStage | None:
        if self.is_final_stage:
            return None
        return self.stages[self._current_index + 1]

    def advance(self) -> bool:
        if self.is_final_stage:
            return False
        self._current_index += 1
        return True

    def maybe_advance(self, metrics: ProgressMetrics) -> bool:
        if not meets_promotion_criteria(self.current_stage, metrics):
            return False
        return self.advance()
