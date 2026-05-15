"""Promotion rules used by the curriculum scheduler."""

from __future__ import annotations

from rocket_landing.application.rl.curriculum.stages import PromotionCriteria
from rocket_landing.application.rl.evaluation.metrics import EvaluationMetrics


def meets_promotion_criteria(
    metrics: EvaluationMetrics,
    criteria: PromotionCriteria,
) -> bool:
    """Return whether the evaluation metrics are strong enough to advance."""

    if metrics.episodes < criteria.min_eval_episodes:
        return False
    if metrics.success_rate < criteria.min_success_rate:
        return False
    if (
        criteria.max_mean_abs_final_dx is not None
        and metrics.mean_abs_final_dx > criteria.max_mean_abs_final_dx
    ):
        return False
    return True
