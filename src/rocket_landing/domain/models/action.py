from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Action:
    """Commanded booster inputs for a single simulation step."""

    throttle: float
    gimbal: float
