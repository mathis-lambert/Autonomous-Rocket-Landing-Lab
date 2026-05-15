"""Observation-space definition for the first RL agent."""

from __future__ import annotations

import numpy as np

OBSERVATION_LABELS: tuple[str, ...] = (
    "dx",
    "z",
    "vx",
    "vz",
    "sin_theta",
    "cos_theta",
    "omega",
    "fuel_ratio",
)

OBSERVATION_LOW = np.array(
    [-10_000.0, 0.0, -1_000.0, -1_000.0, -1.0, -1.0, -100.0, 0.0],
    dtype=np.float32,
)
OBSERVATION_HIGH = np.array(
    [10_000.0, 10_000.0, 1_000.0, 1_000.0, 1.0, 1.0, 100.0, 1.0],
    dtype=np.float32,
)
