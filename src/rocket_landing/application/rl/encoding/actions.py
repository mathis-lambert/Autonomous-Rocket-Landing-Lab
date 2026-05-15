"""Translate agent outputs into domain control commands."""

from __future__ import annotations

import numpy as np

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams


def decode_agent_action(raw_action: np.ndarray, params: RocketParams) -> Action:
    """Convert a 2D RL action into a sanitized domain command."""

    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.shape != (2,):
        raise ValueError(f"expected RL action shape (2,), got {action.shape}")

    throttle = float(np.clip(action[0], 0.0, 1.0))
    gimbal = float(np.clip(action[1], -1.0, 1.0)) * params.max_gimbal
    return Action(throttle=throttle, engine_gimbal=gimbal, aero_steer=0.0)
