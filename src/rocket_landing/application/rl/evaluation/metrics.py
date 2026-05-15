"""Aggregate episode summaries into stable evaluation metrics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rocket_landing.application.rl.specs.episode_types import EpisodeSummary


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregate metrics used for evaluation and curriculum progression."""

    episodes: int
    success_rate: float
    crash_rate: float
    truncation_rate: float
    mean_return: float
    mean_final_dx: float
    mean_abs_final_dx: float
    mean_final_altitude: float
    mean_final_speed: float
    mean_fuel_ratio: float

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-friendly mapping."""

        return asdict(self)


def aggregate_episode_summaries(summaries: list[EpisodeSummary]) -> EvaluationMetrics:
    """Aggregate episode-level results into compact evaluation metrics."""

    if not summaries:
        raise ValueError("at least one episode summary is required for evaluation metrics")

    episodes = len(summaries)
    return EvaluationMetrics(
        episodes=episodes,
        success_rate=sum(1.0 for summary in summaries if summary.landed) / episodes,
        crash_rate=sum(1.0 for summary in summaries if summary.crashed) / episodes,
        truncation_rate=sum(1.0 for summary in summaries if summary.truncated) / episodes,
        mean_return=sum(summary.return_total for summary in summaries) / episodes,
        mean_final_dx=sum(summary.final_dx for summary in summaries) / episodes,
        mean_abs_final_dx=sum(abs(summary.final_dx) for summary in summaries) / episodes,
        mean_final_altitude=sum(summary.final_altitude for summary in summaries) / episodes,
        mean_final_speed=sum(summary.final_speed for summary in summaries) / episodes,
        mean_fuel_ratio=sum(summary.fuel_ratio for summary in summaries) / episodes,
    )
