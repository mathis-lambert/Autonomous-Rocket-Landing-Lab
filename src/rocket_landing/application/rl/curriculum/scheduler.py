"""Stateful curriculum scheduler used by the trainer."""

from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.application.rl.curriculum.progression import meets_promotion_criteria
from rocket_landing.application.rl.curriculum.stages import CurriculumStage
from rocket_landing.application.rl.evaluation.metrics import EvaluationMetrics


@dataclass(slots=True)
class CurriculumScheduler:
    """Track the active curriculum stage and move forward when criteria are met."""

    stages: tuple[CurriculumStage, ...]
    _current_index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("curriculum scheduler requires at least one stage")

    @property
    def current_stage(self) -> CurriculumStage:
        """Return the active curriculum stage."""

        return self.stages[self._current_index]

    @property
    def current_index(self) -> int:
        """Return the zero-based index of the active stage."""

        return self._current_index

    @property
    def is_final_stage(self) -> bool:
        """Return whether the scheduler already reached the last stage."""

        return self._current_index >= len(self.stages) - 1

    def maybe_advance(self, metrics: EvaluationMetrics) -> bool:
        """Advance to the next stage when the current criteria are met."""

        if self.is_final_stage:
            return False
        if not meets_promotion_criteria(metrics, self.current_stage.promotion):
            return False
        self._current_index += 1
        return True
