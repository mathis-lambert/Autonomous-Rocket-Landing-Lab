"""Reward shaping configuration for RL training."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Dense shaping weights applied every environment step."""

    dx: float = 0.12
    z: float = 0.08
    vx: float = 0.12
    vz: float = 0.15
    tilt: float = 0.20
    omega: float = 0.06
    throttle: float = 0.005
    gimbal: float = 0.002
    step: float = 0.15
    dx_progress: float = 0.5
    z_progress: float = 1.0


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Terminal and dense reward configuration."""

    weights: RewardWeights = field(default_factory=RewardWeights)
    landing_bonus: float = 2_000.0
    crash_penalty: float = 800.0
