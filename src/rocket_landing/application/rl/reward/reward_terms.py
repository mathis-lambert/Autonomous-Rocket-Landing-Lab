"""Small reward helpers kept separate from the environment orchestration."""

from __future__ import annotations

import math

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


def wrapped_angle_error(theta: float) -> float:
    """Return the absolute wrapped angle error in radians."""

    return abs(math.atan2(math.sin(theta), math.cos(theta)))


def distance_penalty(state: State, params: RocketParams, weight: float) -> float:
    """Penalize lateral distance from the target pad."""

    return -weight * abs(state.horizontal_error(params.target_x))


def altitude_penalty(state: State, weight: float) -> float:
    """Penalize staying high above the landing zone."""

    return -weight * max(0.0, state.z)


def distance_progress_reward(
    previous_state: State,
    current_state: State,
    params: RocketParams,
    weight: float,
) -> float:
    """Reward reducing the lateral target error from one step to the next."""

    previous_error = abs(previous_state.horizontal_error(params.target_x))
    current_error = abs(current_state.horizontal_error(params.target_x))
    return weight * (previous_error - current_error)


def descent_progress_reward(previous_state: State, current_state: State, weight: float) -> float:
    """Reward making downward progress toward touchdown."""

    return weight * max(0.0, previous_state.z - current_state.z)


def horizontal_velocity_penalty(state: State, weight: float) -> float:
    """Penalize lateral speed."""

    return -weight * abs(state.vx)


def vertical_velocity_penalty(state: State, weight: float) -> float:
    """Penalize descent or climb rate."""

    return -weight * abs(state.vz)


def tilt_penalty(state: State, weight: float) -> float:
    """Penalize attitude error relative to upright flight."""

    return -weight * wrapped_angle_error(state.theta)


def angular_velocity_penalty(state: State, weight: float) -> float:
    """Penalize rotation rate."""

    return -weight * abs(state.omega)


def throttle_penalty(action: Action, weight: float) -> float:
    """Penalize control effort through throttle usage."""

    return -weight * action.throttle


def gimbal_penalty(action: Action, params: RocketParams, weight: float) -> float:
    """Penalize control effort through engine gimbal usage."""

    if params.max_gimbal <= 0.0:
        return 0.0
    return -weight * abs(action.engine_gimbal / params.max_gimbal)


def step_penalty(weight: float) -> float:
    """Penalize long episodes to discourage hovering and stalling."""

    return -abs(weight)
