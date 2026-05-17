"""Shared helpers for training and evaluation campaigns."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from rocket_landing.rl.curriculum import CurriculumScheduler
from rocket_landing.rl.env import RocketLanderEnv
from rocket_landing.rl.evaluation import evaluate_policy
from rocket_landing.rl.sb3 import load_sac_model
from rocket_landing.rl.types import (
    CurriculumStage,
    EnvConfig,
    EvaluationConfig,
    RewardConfig,
    TrainingConfig,
)


def parse_seed_list(raw: str) -> list[int]:
    """Parse a comma-separated seed list."""

    seeds: list[int] = []
    for item in raw.split(","):
        token = item.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def slice_curriculum_from_stage(
    stage_name: str,
    stages: tuple[CurriculumStage, ...],
) -> tuple[CurriculumStage, ...]:
    """Return the curriculum suffix that starts at the requested stage."""

    names = [stage.name for stage in stages]
    try:
        start_index = names.index(stage_name)
    except ValueError as exc:
        available = ", ".join(names)
        raise ValueError(
            f"unknown curriculum start stage '{stage_name}', available: {available}"
        ) from exc
    return stages[start_index:]


def build_stage_environment(
    simulation_config: Any,
    *,
    stage: CurriculumStage | None = None,
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> RocketLanderEnv:
    """Create one environment and optionally align it to a curriculum stage."""

    env = RocketLanderEnv(
        simulation_config.params,
        env_config=env_config or EnvConfig(),
        reward_config=reward_config or RewardConfig(),
        default_initial_state=simulation_config.initial_state,
    )
    if stage is not None:
        env.apply_curriculum_stage(stage)
    return env


def build_training_env_factory(
    simulation_config: Any,
    *,
    fixed_stage: CurriculumStage | None = None,
    env_config: EnvConfig | None = None,
    reward_config: RewardConfig | None = None,
) -> Callable[[], RocketLanderEnv]:
    """Create a stable env factory for vectorized training."""

    local_env_config = env_config or EnvConfig()
    local_reward_config = reward_config or RewardConfig()

    def env_factory() -> RocketLanderEnv:
        return build_stage_environment(
            simulation_config,
            stage=fixed_stage,
            env_config=local_env_config,
            reward_config=local_reward_config,
        )

    return env_factory


def build_scheduler(
    *,
    stages: tuple[CurriculumStage, ...],
    no_curriculum: bool,
    resume_from: Path | None,
    resume_stage: str | None,
) -> CurriculumScheduler | None:
    """Build the optional curriculum scheduler for one training run."""

    if no_curriculum:
        return None
    curriculum_stages = stages
    if resume_from is not None and resume_stage is not None:
        curriculum_stages = slice_curriculum_from_stage(resume_stage, stages)
    return CurriculumScheduler(curriculum_stages)


def apply_training_overrides(
    config: TrainingConfig,
    *,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    gradient_steps: int | None = None,
    learning_starts: int | None = None,
    buffer_size: int | None = None,
) -> TrainingConfig:
    """Return a training config with selected SAC overrides applied."""

    return replace(
        config,
        sac=replace(
            config.sac,
            learning_rate=config.sac.learning_rate if learning_rate is None else learning_rate,
            batch_size=config.sac.batch_size if batch_size is None else batch_size,
            gradient_steps=(
                config.sac.gradient_steps if gradient_steps is None else gradient_steps
            ),
            learning_starts=(
                config.sac.learning_starts if learning_starts is None else learning_starts
            ),
            buffer_size=config.sac.buffer_size if buffer_size is None else buffer_size,
        ),
    )


def benchmark_saved_model(
    *,
    model_path: Path,
    simulation_config: Any,
    stage: CurriculumStage,
    episodes: int,
    seed: int,
    deterministic: bool = True,
    device: str = "auto",
) -> Any:
    """Benchmark one saved model against one curriculum stage."""

    env = build_stage_environment(simulation_config, stage=stage)
    model = load_sac_model(model_path, device=device)
    report = evaluate_policy(
        env,
        lambda obs: model.predict(obs, deterministic=deterministic)[0],
        config=EvaluationConfig(
            episodes=episodes,
            deterministic=deterministic,
            seed=seed,
        ),
    )
    env.close()
    return report


def write_campaign_summary(
    *,
    campaign_dir: Path,
    campaign_name: str,
    scenario_name: str,
    seeds: list[int],
    benchmark_stage_name: str,
    benchmark_success: list[float],
    benchmark_return: list[float],
    seed_runs: list[dict[str, object]],
) -> Path:
    """Persist aggregate campaign results and return the summary path."""

    payload = {
        "campaign_name": campaign_name,
        "scenario": scenario_name,
        "seeds": seeds,
        "benchmark_stage": benchmark_stage_name,
        "benchmark_mean_success_rate": mean(benchmark_success),
        "benchmark_std_success_rate": (
            0.0 if len(benchmark_success) == 1 else pstdev(benchmark_success)
        ),
        "benchmark_mean_return": mean(benchmark_return),
        "benchmark_std_return": 0.0 if len(benchmark_return) == 1 else pstdev(benchmark_return),
        "runs": seed_runs,
    }
    path = campaign_dir / "campaign_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
