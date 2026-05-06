from __future__ import annotations

import math

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


class BoosterGeometry:
    """Computes simple geometric primitives for the booster."""

    def __init__(self, params: RocketParams) -> None:
        self._params = params

    def segment_endpoints(self, state: State) -> tuple[tuple[float, float], tuple[float, float]]:
        half_length = self._params.length / 2.0
        dx = math.sin(state.theta) * half_length
        dz = math.cos(state.theta) * half_length
        bottom = (state.x - dx, state.z - dz)
        top = (state.x + dx, state.z + dz)
        return bottom, top
