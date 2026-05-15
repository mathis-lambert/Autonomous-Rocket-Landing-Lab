"""Build compact observation vectors from simulator state."""

from __future__ import annotations

import math

import numpy as np

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


def fuel_ratio_for(state: State, params: RocketParams) -> float:
    """Return normalized remaining propellant in the inclusive range ``[0, 1]``."""

    if params.initial_fuel <= 0.0:
        return 0.0
    return min(max(state.fuel / params.initial_fuel, 0.0), 1.0)


def build_observation(state: State, params: RocketParams) -> np.ndarray:
    """Encode the current world state as a compact RL observation vector."""

    return np.array(
        [
            state.horizontal_error(params.target_x),
            state.z,
            state.vx,
            state.vz,
            math.sin(state.theta),
            math.cos(state.theta),
            state.omega,
            fuel_ratio_for(state, params),
        ],
        dtype=np.float32,
    )
