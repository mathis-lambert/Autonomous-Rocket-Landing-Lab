"""Reward computation for RL training."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State
from rocket_landing.rl.types import RewardConfig


def _wrapped_angle_error(theta: float) -> float:
    return abs(math.atan2(math.sin(theta), math.cos(theta)))


def _fuel_ratio_for(state: State, params: RocketParams) -> float:
    if params.initial_fuel <= 0.0:
        return 0.0
    return min(max(state.fuel / params.initial_fuel, 0.0), 1.0)


def _excess(value: float, limit: float) -> float:
    return max(0.0, abs(value) - limit)


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Individual terms contributing to the scalar reward."""

    dx: float
    z: float
    dx_progress: float
    z_progress: float
    guidance_dx_progress: float
    guidance_vx: float
    vx: float
    vz: float
    tilt: float
    omega: float
    near_ground_vx: float
    near_ground_vz: float
    near_ground_tilt: float
    near_ground_omega: float
    throttle: float
    gimbal: float
    step: float
    terminal: float

    @property
    def total(self) -> float:
        return (
            self.dx
            + self.z
            + self.dx_progress
            + self.z_progress
            + self.guidance_dx_progress
            + self.guidance_vx
            + self.vx
            + self.vz
            + self.tilt
            + self.omega
            + self.near_ground_vx
            + self.near_ground_vz
            + self.near_ground_tilt
            + self.near_ground_omega
            + self.throttle
            + self.gimbal
            + self.step
            + self.terminal
        )

    def as_dict(self) -> dict[str, float]:
        values = asdict(self)
        values["total"] = self.total
        return values


def compute_reward(
    *,
    action: Action,
    previous_state: State,
    result: StepResult,
    params: RocketParams,
    config: RewardConfig,
    truncated: bool = False,
) -> RewardBreakdown:
    """Compute one-step reward from transition and control action."""

    state = result.impact_state or result.state
    weights = config.weights
    previous_dx = abs(previous_state.horizontal_error(params.target_x))
    current_dx = abs(state.horizontal_error(params.target_x))
    ground_clearance = max(0.0, state.z - (params.length * 0.5))
    near_ground = 0.0
    if config.near_ground_altitude > 0.0:
        near_ground = max(0.0, 1.0 - (ground_clearance / config.near_ground_altitude))
    guidance = 0.0
    if config.guidance_altitude > 0.0:
        guidance = min(1.0, ground_clearance / config.guidance_altitude)
    terminal = 0.0
    if result.landed:
        terminal = config.landing_bonus
    elif result.crashed:
        terminal = -config.crash_penalty
    elif truncated:
        terminal = -config.truncation_penalty

    return RewardBreakdown(
        dx=-weights.dx * current_dx,
        z=-weights.z * max(0.0, state.z),
        dx_progress=weights.dx_progress * (previous_dx - current_dx),
        z_progress=weights.z_progress * max(0.0, previous_state.z - state.z),
        guidance_dx_progress=guidance * weights.guidance_dx_progress * (previous_dx - current_dx),
        guidance_vx=-guidance * weights.guidance_vx * abs(state.vx),
        vx=-weights.vx * abs(state.vx),
        vz=-weights.vz * abs(state.vz),
        tilt=-weights.tilt * _wrapped_angle_error(state.theta),
        omega=-weights.omega * abs(state.omega),
        near_ground_vx=-near_ground
        * weights.near_ground_vx
        * _excess(state.vx, params.max_landing_vx),
        near_ground_vz=-near_ground
        * weights.near_ground_vz
        * _excess(state.vz, params.max_landing_vz),
        near_ground_tilt=(
            -near_ground
            * weights.near_ground_tilt
            * max(0.0, _wrapped_angle_error(state.theta) - params.max_landing_theta)
        ),
        near_ground_omega=(
            -near_ground
            * weights.near_ground_omega
            * _excess(state.omega, params.max_landing_omega)
        ),
        throttle=-weights.throttle * action.throttle,
        gimbal=0.0
        if params.max_gimbal <= 0.0
        else -weights.gimbal * abs(action.engine_gimbal / params.max_gimbal),
        step=-abs(weights.step),
        terminal=terminal,
    )


def build_episode_summary(
    *,
    step_count: int,
    elapsed_time: float,
    episode_return: float,
    result: StepResult,
    params: RocketParams,
    truncated: bool,
    truncation_reason: str | None = None,
) -> dict[str, float | bool | int | str | None]:
    """Build compact end-of-episode telemetry."""

    state = result.impact_state or result.state
    return {
        "steps": step_count,
        "elapsed_time": elapsed_time,
        "return_total": episode_return,
        "landed": result.landed,
        "crashed": result.crashed,
        "truncated": truncated,
        "truncation_reason": truncation_reason,
        "final_dx": state.horizontal_error(params.target_x),
        "final_altitude": state.z,
        "final_vx": state.vx,
        "final_vz": state.vz,
        "final_theta": state.theta,
        "final_omega": state.omega,
        "final_speed": state.speed,
        "fuel_ratio": _fuel_ratio_for(state, params),
    }
