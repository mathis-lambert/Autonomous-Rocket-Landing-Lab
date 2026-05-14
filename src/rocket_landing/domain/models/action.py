"""Domain command objects describing how the booster is actuated."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Action:
    """Commanded booster inputs for a single simulation step."""

    throttle: float
    """Normalized throttle command in the inclusive range ``[0, 1]``."""

    engine_gimbal: float
    """Requested engine gimbal angle in radians."""

    aero_steer: float = 0.0
    """Normalized aerodynamic steering command in the inclusive range ``[-1, 1]``."""

    @classmethod
    def neutral(cls) -> Action:
        """Return the fully neutral control command."""

        return cls(throttle=0.0, engine_gimbal=0.0, aero_steer=0.0)
