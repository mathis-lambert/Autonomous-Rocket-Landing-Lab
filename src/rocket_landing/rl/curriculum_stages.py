"""Curriculum stage definitions for progressive RL training."""

from __future__ import annotations

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.rl.types import (
    CurriculumStage,
    InitialStateDistribution,
    PromotionCriteria,
    RewardConfig,
    RewardWeights,
    TruncationConfig,
    UniformRange,
    default_initial_state_distribution,
)


def _touchdown_reward_config() -> RewardConfig:
    return RewardConfig(
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
        truncation_penalty=900.0,
    )


def _guided_descent_reward_config() -> RewardConfig:
    return RewardConfig(
        weights=RewardWeights(
            dx=0.16,
            z=0.04,
            vx=0.16,
            vz=0.14,
            tilt=0.22,
            omega=0.07,
            throttle=0.004,
            gimbal=0.002,
            step=0.10,
            dx_progress=0.80,
            z_progress=0.70,
            guidance_dx_progress=1.25,
            guidance_vx=0.25,
            near_ground_vx=0.35,
            near_ground_vz=0.55,
            near_ground_tilt=0.80,
            near_ground_omega=0.25,
        ),
        landing_bonus=2_500.0,
        crash_penalty=1_000.0,
        truncation_penalty=1_100.0,
        guidance_altitude=70.0,
        near_ground_altitude=8.0,
    )


def _full_envelope_reward_config() -> RewardConfig:
    return RewardConfig(
        weights=RewardWeights(
            dx=0.18,
            z=0.03,
            vx=0.18,
            vz=0.12,
            tilt=0.24,
            omega=0.08,
            throttle=0.003,
            gimbal=0.002,
            step=0.08,
            dx_progress=1.00,
            z_progress=0.85,
            guidance_dx_progress=1.80,
            guidance_vx=0.35,
            near_ground_vx=0.40,
            near_ground_vz=0.60,
            near_ground_tilt=0.85,
            near_ground_omega=0.30,
        ),
        landing_bonus=2_500.0,
        crash_penalty=1_100.0,
        truncation_penalty=1_200.0,
        guidance_altitude=90.0,
        near_ground_altitude=8.0,
    )


def build_default_curriculum(params: RocketParams) -> tuple[CurriculumStage, ...]:
    """Return a progressive curriculum from easy touchdown to full envelope."""

    touchdown_reward = _touchdown_reward_config()
    guided_descent_reward = _guided_descent_reward_config()
    full_envelope_reward = _full_envelope_reward_config()
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
                min_eval_episodes=50,
                max_mean_abs_final_dx=params.max_landing_x * 0.25,
                max_crash_rate=0.10,
                max_truncation_rate=0.10,
                max_mean_abs_final_vx=0.80,
                max_mean_abs_final_vz=2.50,
                max_mean_abs_final_theta=0.05,
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
                min_success_rate=0.80,
                min_eval_episodes=50,
                max_mean_abs_final_dx=params.max_landing_x * 0.5,
                max_crash_rate=0.10,
                max_truncation_rate=0.10,
                max_mean_abs_final_vx=0.90,
                max_mean_abs_final_vz=2.60,
                max_mean_abs_final_theta=0.06,
            ),
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
                min_success_rate=0.75,
                min_eval_episodes=50,
                max_mean_abs_final_dx=params.max_landing_x,
                max_crash_rate=0.15,
                max_truncation_rate=0.15,
                max_mean_abs_final_vx=1.10,
                max_mean_abs_final_vz=2.80,
                max_mean_abs_final_theta=0.07,
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
                min_success_rate=0.70,
                min_eval_episodes=50,
                max_mean_abs_final_dx=params.max_landing_x * 1.5,
                max_crash_rate=0.15,
                max_truncation_rate=0.15,
                max_mean_abs_final_vx=1.20,
                max_mean_abs_final_vz=3.00,
                max_mean_abs_final_theta=0.08,
            ),
            reward_config=guided_descent_reward,
        ),
        CurriculumStage(
            name="high_descent",
            reset_distribution=InitialStateDistribution(
                dx=UniformRange(-18.0, 18.0),
                z=UniformRange(60.0, 100.0),
                vx=UniformRange(-4.0, 4.0),
                vz=UniformRange(-10.0, -4.0),
                theta=UniformRange(-0.07, 0.07),
                omega=UniformRange(-0.05, 0.05),
                fuel_ratio=UniformRange(0.85, 1.0),
            ),
            truncation=TruncationConfig(max_abs_dx=120.0, max_altitude=220.0),
            promotion=PromotionCriteria(
                min_success_rate=0.70,
                min_eval_episodes=50,
                max_mean_abs_final_dx=params.max_landing_x * 2.0,
                max_crash_rate=0.20,
                max_truncation_rate=0.20,
                max_mean_abs_final_vx=1.40,
                max_mean_abs_final_vz=3.20,
                max_mean_abs_final_theta=0.09,
            ),
            reward_config=guided_descent_reward,
        ),
        CurriculumStage(
            name="wide_recovery",
            reset_distribution=InitialStateDistribution(
                dx=UniformRange(-22.0, 22.0),
                z=UniformRange(70.0, 120.0),
                vx=UniformRange(-4.5, 4.5),
                vz=UniformRange(-12.0, -5.0),
                theta=UniformRange(-0.08, 0.08),
                omega=UniformRange(-0.06, 0.06),
                fuel_ratio=UniformRange(0.85, 1.0),
            ),
            truncation=TruncationConfig(max_abs_dx=170.0, max_altitude=280.0),
            promotion=PromotionCriteria(
                min_success_rate=0.65,
                min_eval_episodes=50,
                max_mean_abs_final_dx=params.max_landing_x * 2.0,
                max_crash_rate=0.20,
                max_truncation_rate=0.25,
                max_mean_abs_final_vx=1.50,
                max_mean_abs_final_vz=3.40,
                max_mean_abs_final_theta=0.09,
            ),
            reward_config=guided_descent_reward,
        ),
        CurriculumStage(
            name="full_envelope_nominal_fuel",
            reset_distribution=InitialStateDistribution(
                dx=UniformRange(-25.0, 25.0),
                z=UniformRange(80.0, 140.0),
                vx=UniformRange(-6.0, 6.0),
                vz=UniformRange(-18.0, -7.0),
                theta=UniformRange(-0.10, 0.10),
                omega=UniformRange(-0.08, 0.08),
                fuel_ratio=UniformRange(0.85, 1.0),
            ),
            truncation=TruncationConfig(max_abs_dx=220.0, max_altitude=420.0),
            promotion=PromotionCriteria(
                min_success_rate=0.60,
                min_eval_episodes=50,
                max_mean_abs_final_dx=params.max_landing_x * 2.5,
                max_crash_rate=0.25,
                max_truncation_rate=0.25,
                max_mean_abs_final_vx=1.60,
                max_mean_abs_final_vz=3.80,
                max_mean_abs_final_theta=0.10,
            ),
            reward_config=full_envelope_reward,
        ),
        CurriculumStage(
            name="full_envelope",
            reset_distribution=default_initial_state_distribution(),
            truncation=TruncationConfig(max_abs_dx=250.0, max_altitude=500.0),
            promotion=PromotionCriteria(
                min_success_rate=0.70,
                min_eval_episodes=50,
                max_mean_abs_final_dx=params.max_landing_x * 2.5,
                max_crash_rate=0.25,
                max_truncation_rate=0.25,
                max_mean_abs_final_vx=1.70,
                max_mean_abs_final_vz=4.00,
                max_mean_abs_final_theta=0.10,
            ),
            reward_config=full_envelope_reward,
        ),
    )


def find_stage_by_name(stages: tuple[CurriculumStage, ...], name: str) -> CurriculumStage:
    """Resolve a curriculum stage by name."""

    for stage in stages:
        if stage.name == name:
            return stage
    available = ", ".join(stage.name for stage in stages)
    raise ValueError(f"unknown stage '{name}', available: {available}")
