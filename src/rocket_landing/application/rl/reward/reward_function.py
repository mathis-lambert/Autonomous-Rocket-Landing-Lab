"""Aggregate dense shaping and terminal events into a scalar reward."""

from __future__ import annotations

from rocket_landing.application.rl.config.reward_config import RewardConfig
from rocket_landing.application.rl.reward.reward_breakdown import RewardBreakdown
from rocket_landing.application.rl.reward.reward_terms import (
    altitude_penalty,
    angular_velocity_penalty,
    descent_progress_reward,
    distance_penalty,
    distance_progress_reward,
    gimbal_penalty,
    horizontal_velocity_penalty,
    step_penalty,
    throttle_penalty,
    tilt_penalty,
    vertical_velocity_penalty,
)
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State


def compute_reward(
    *,
    action: Action,
    previous_state: State,
    result: StepResult,
    params: RocketParams,
    config: RewardConfig,
) -> RewardBreakdown:
    """Compute reward shaping for one transition."""

    state = result.impact_state or result.state
    weights = config.weights
    terminal = 0.0
    if result.landed:
        terminal = config.landing_bonus
    elif result.crashed:
        terminal = -config.crash_penalty

    return RewardBreakdown(
        dx=distance_penalty(state, params, weights.dx),
        z=altitude_penalty(state, weights.z),
        dx_progress=distance_progress_reward(previous_state, state, params, weights.dx_progress),
        z_progress=descent_progress_reward(previous_state, state, weights.z_progress),
        vx=horizontal_velocity_penalty(state, weights.vx),
        vz=vertical_velocity_penalty(state, weights.vz),
        tilt=tilt_penalty(state, weights.tilt),
        omega=angular_velocity_penalty(state, weights.omega),
        throttle=throttle_penalty(action, weights.throttle),
        gimbal=gimbal_penalty(action, params, weights.gimbal),
        step=step_penalty(weights.step),
        terminal=terminal,
    )
