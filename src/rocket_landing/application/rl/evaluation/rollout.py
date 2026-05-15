"""Environment rollout helpers independent from any RL framework."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.application.rl.specs.episode_types import EpisodeSummary

PolicyFn = Callable[[np.ndarray], np.ndarray]


def run_episode(
    env: RocketLanderEnv,
    policy_fn: PolicyFn,
    *,
    seed: int | None = None,
) -> EpisodeSummary:
    """Run one episode and return its final summary."""

    observation, _info = env.reset(seed=seed)
    while True:
        action = np.asarray(policy_fn(observation), dtype=np.float32)
        observation, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            episode_info = info.get("episode")
            if not isinstance(episode_info, dict):
                raise RuntimeError("episode summary missing from terminal environment info")
            return EpisodeSummary(**episode_info)
