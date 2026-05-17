"""Policy rollout and aggregate evaluation helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from rocket_landing.rl.env import RocketLanderEnv
from rocket_landing.rl.types import EvaluationConfig

PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """Compact end-of-episode telemetry."""

    steps: int
    elapsed_time: float
    return_total: float
    landed: bool
    crashed: bool
    truncated: bool
    truncation_reason: str | None
    final_dx: float
    final_altitude: float
    final_vx: float
    final_vz: float
    final_theta: float
    final_omega: float
    final_speed: float
    fuel_ratio: float

    def as_dict(self) -> dict[str, float | bool | int | str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    """Aggregated metrics over several episodes."""

    episodes: int
    success_rate: float
    crash_rate: float
    truncation_rate: float
    max_steps_truncation_rate: float
    lateral_bounds_truncation_rate: float
    altitude_bounds_truncation_rate: float
    mean_return: float
    mean_final_dx: float
    mean_abs_final_dx: float
    mean_final_altitude: float
    mean_abs_final_vx: float
    mean_abs_final_vz: float
    mean_abs_final_theta: float
    mean_abs_final_omega: float
    mean_final_speed: float
    mean_fuel_ratio: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Detailed report containing raw episodes and aggregate metrics."""

    episodes: tuple[EpisodeSummary, ...]
    metrics: EvaluationMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.as_dict(),
            "episodes": [episode.as_dict() for episode in self.episodes],
        }


def run_episode(
    env: RocketLanderEnv,
    policy_fn: PolicyFn,
    *,
    seed: int | None = None,
) -> EpisodeSummary:
    """Run one policy episode and return its summary."""

    observation, _info = env.reset(seed=seed)
    while True:
        action = np.asarray(policy_fn(observation), dtype=np.float32)
        observation, _reward, terminated, truncated, info = env.step(action)
        if not (terminated or truncated):
            continue
        episode_info = info.get("episode")
        if not isinstance(episode_info, dict):
            raise RuntimeError("episode summary missing from terminal info payload")
        return EpisodeSummary(**episode_info)


def aggregate_episode_summaries(episodes: list[EpisodeSummary]) -> EvaluationMetrics:
    """Aggregate episode list into stable scalar metrics."""

    if not episodes:
        raise ValueError("at least one episode is required")
    count = len(episodes)
    return EvaluationMetrics(
        episodes=count,
        success_rate=sum(1.0 for item in episodes if item.landed) / count,
        crash_rate=sum(1.0 for item in episodes if item.crashed) / count,
        truncation_rate=sum(1.0 for item in episodes if item.truncated) / count,
        max_steps_truncation_rate=sum(
            1.0 for item in episodes if item.truncation_reason == "max_episode_steps"
        )
        / count,
        lateral_bounds_truncation_rate=sum(
            1.0 for item in episodes if item.truncation_reason == "max_abs_dx"
        )
        / count,
        altitude_bounds_truncation_rate=sum(
            1.0 for item in episodes if item.truncation_reason == "max_altitude"
        )
        / count,
        mean_return=sum(item.return_total for item in episodes) / count,
        mean_final_dx=sum(item.final_dx for item in episodes) / count,
        mean_abs_final_dx=sum(abs(item.final_dx) for item in episodes) / count,
        mean_final_altitude=sum(item.final_altitude for item in episodes) / count,
        mean_abs_final_vx=sum(abs(item.final_vx) for item in episodes) / count,
        mean_abs_final_vz=sum(abs(item.final_vz) for item in episodes) / count,
        mean_abs_final_theta=sum(abs(item.final_theta) for item in episodes) / count,
        mean_abs_final_omega=sum(abs(item.final_omega) for item in episodes) / count,
        mean_final_speed=sum(item.final_speed for item in episodes) / count,
        mean_fuel_ratio=sum(item.fuel_ratio for item in episodes) / count,
    )


def evaluate_policy(
    env: RocketLanderEnv,
    policy_fn: PolicyFn,
    *,
    config: EvaluationConfig | None = None,
) -> EvaluationReport:
    """Evaluate one policy against the provided environment."""

    evaluation_config = config or EvaluationConfig()
    episodes = [
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
        episodes=tuple(episodes),
        metrics=aggregate_episode_summaries(episodes),
    )
