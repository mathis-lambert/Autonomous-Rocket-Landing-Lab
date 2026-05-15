"""Action-space definition for the first RL agent."""

from __future__ import annotations

import numpy as np

ACTION_LABELS: tuple[str, ...] = ("throttle", "gimbal")
ACTION_LOW = np.array([0.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)
