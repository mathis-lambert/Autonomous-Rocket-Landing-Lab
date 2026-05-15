"""Evaluation orchestration for policies acting on the RL environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rocket_landing.application.rl.config import EvaluationConfig
from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.application.rl.evaluation.metrics import (
    EvaluationMetrics,
    aggregate_episode_summaries,
)
from rocket_landing.application.rl.evaluation.rollout import PolicyFn, run_episode
from rocket_landing.application.rl.specs.episode_types import EpisodeSummary


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Detailed evaluation report with both raw episodes and aggregate metrics."""

    episodes: tuple[EpisodeSummary, ...]
    metrics: EvaluationMetrics

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly mapping."""

        return {
            "metrics": self.metrics.as_dict(),
            "episodes": [asdict(summary) for summary in self.episodes],
        }


def evaluate_policy(
    env: RocketLanderEnv,
    policy_fn: PolicyFn,
    *,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Evaluate a policy over several fresh episodes."""

    evaluation_config = config or EvaluationConfig()
    summaries = [
        run_episode(
            env,
            policy_fn,
            seed=None
            if evaluation_config.seed is None
            else evaluation_config.seed + episode_index,
        )
        for episode_index in range(evaluation_config.episodes)
    ]
    return EvaluationReport(
        episodes=tuple(summaries),
        metrics=aggregate_episode_summaries(summaries),
    )
