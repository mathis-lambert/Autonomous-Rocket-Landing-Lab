"""Curriculum stage definitions for progressively harder training."""

from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.application.rl.config import RewardConfig, RewardWeights
from rocket_landing.application.rl.reset.distributions import (
    InitialStateDistribution,
    UniformRange,
    default_initial_state_distribution,
)
from rocket_landing.application.rl.termination import TruncationConfig
from rocket_landing.domain.models.params import RocketParams


@dataclass(frozen=True, slots=True)
class PromotionCriteria:
    """Conditions that must be met before moving to the next curriculum stage."""

    min_success_rate: float
    min_eval_episodes: int = 20
    max_mean_abs_final_dx: float | None = None


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """Bundle the runtime overrides for one curriculum stage."""

    name: str
    reset_distribution: InitialStateDistribution
    truncation: TruncationConfig
    promotion: PromotionCriteria
    reward_config: RewardConfig | None = None


def build_default_curriculum(params: RocketParams) -> tuple[CurriculumStage, ...]:
    """Return a conservative default curriculum for early SAC training."""

    touchdown_reward = RewardConfig(
        weights=RewardWeights(
            dx=0.10,
            z=0.08,
            vx=0.15,
            vz=0.20,
            tilt=0.25,
            omega=0.08,
            throttle=0.005,
            gimbal=0.002,
            step=0.12,
            dx_progress=0.25,
            z_progress=0.60,
        ),
        landing_bonus=2_500.0,
        crash_penalty=1_000.0,
    )
    stabilize_reward = RewardConfig()
    return (
        CurriculumStage(
            name="touchdown",
            reset_distribution=InitialStateDistribution(
                dx=UniformRange(-0.75, 0.75),
                z=UniformRange(13.0, 17.0),
                vx=UniformRange(-0.2, 0.2),
                vz=UniformRange(-1.5, -0.2),
                theta=UniformRange(-0.008, 0.008),
                omega=UniformRange(-0.01, 0.01),
                fuel_ratio=UniformRange(1.0, 1.0),
            ),
            truncation=TruncationConfig(max_abs_dx=8.0, max_altitude=25.0),
            promotion=PromotionCriteria(
                min_success_rate=0.85,
                min_eval_episodes=20,
                max_mean_abs_final_dx=0.5,
            ),
            reward_config=touchdown_reward,
        ),
        CurriculumStage(
            name="stabilize",
            reset_distribution=InitialStateDistribution(
                dx=UniformRange(-1.5, 1.5),
                z=UniformRange(14.0, 22.0),
                vx=UniformRange(-0.5, 0.5),
                vz=UniformRange(-3.0, -0.5),
                theta=UniformRange(-0.015, 0.015),
                omega=UniformRange(-0.015, 0.015),
                fuel_ratio=UniformRange(1.0, 1.0),
            ),
            truncation=TruncationConfig(max_abs_dx=15.0, max_altitude=40.0),
            promotion=PromotionCriteria(
                min_success_rate=0.75,
                min_eval_episodes=20,
                max_mean_abs_final_dx=1.0,
            ),
            reward_config=stabilize_reward,
        ),
        CurriculumStage(
            name="offset_recovery",
            reset_distribution=InitialStateDistribution(
                dx=UniformRange(-4.0, 4.0),
                z=UniformRange(20.0, 35.0),
                vx=UniformRange(-1.0, 1.0),
                vz=UniformRange(-4.0, -1.0),
                theta=UniformRange(-0.025, 0.025),
                omega=UniformRange(-0.02, 0.02),
                fuel_ratio=UniformRange(0.98, 1.0),
            ),
            truncation=TruncationConfig(max_abs_dx=25.0, max_altitude=60.0),
            promotion=PromotionCriteria(
                min_success_rate=0.65,
                min_eval_episodes=20,
                max_mean_abs_final_dx=2.5,
            ),
        ),
        CurriculumStage(
            name="fast_descent",
            reset_distribution=InitialStateDistribution(
                dx=UniformRange(-10.0, 10.0),
                z=UniformRange(35.0, 60.0),
                vx=UniformRange(-2.0, 2.0),
                vz=UniformRange(-6.0, -2.0),
                theta=UniformRange(-0.04, 0.04),
                omega=UniformRange(-0.03, 0.03),
                fuel_ratio=UniformRange(0.92, 1.0),
            ),
            truncation=TruncationConfig(max_abs_dx=50.0, max_altitude=120.0),
            promotion=PromotionCriteria(
                min_success_rate=0.60,
                min_eval_episodes=20,
                max_mean_abs_final_dx=params.max_landing_x * 1.5,
            ),
        ),
        CurriculumStage(
            name="full_envelope",
            reset_distribution=default_initial_state_distribution(),
            truncation=TruncationConfig(max_abs_dx=250.0, max_altitude=500.0),
            promotion=PromotionCriteria(
                min_success_rate=1.0,
                min_eval_episodes=20,
            ),
        ),
    )
