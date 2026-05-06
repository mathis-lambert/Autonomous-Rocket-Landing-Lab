"""Small geometry helpers used by analysis and renderers.

The physics core itself does not need explicit geometry primitives for the
current 2D model, but renderers and plotting tools do.
"""

from __future__ import annotations

import math

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


class BoosterGeometry:
    """Computes simple geometric primitives for the booster.

    The current representation treats the booster as a line segment centered on
    its center of mass, aligned with ``theta``.
    """

    def __init__(self, params: RocketParams) -> None:
        self._params = params

    def segment_endpoints(self, state: State) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return bottom and top endpoints for the current booster attitude.

        Args:
            state: Current booster state.

        Returns:
            A pair ``(bottom, top)`` where each point is an ``(x, z)`` tuple in
            world-space meters.
        """

        half_length = self._params.length / 2.0
        dx = math.sin(state.theta) * half_length
        dz = math.cos(state.theta) * half_length
        bottom = (state.x - dx, state.z - dz)
        top = (state.x + dx, state.z + dz)
        return bottom, top
