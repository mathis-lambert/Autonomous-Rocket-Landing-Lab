"""Domain command objects describing how the booster is actuated."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Action:
    """Commanded booster inputs for a single simulation step."""

    throttle: float
    """Normalized throttle command in the inclusive range ``[0, 1]``."""

    gimbal: float
    """Requested engine gimbal angle in radians."""
