"""Environment-level configuration for RL rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.application.rl.reset.distributions import (
    InitialStateDistribution,
    default_initial_state_distribution,
)
from rocket_landing.application.rl.termination.truncation_rules import TruncationConfig


@dataclass(frozen=True, slots=True)
class RLEnvConfig:
    """Configure the RL environment wrapper around the simulator."""

    dt: float = 0.02
    max_episode_steps: int = 1_000
    truncation: TruncationConfig = field(default_factory=TruncationConfig)
    reset_distribution: InitialStateDistribution = field(
        default_factory=default_initial_state_distribution
    )
