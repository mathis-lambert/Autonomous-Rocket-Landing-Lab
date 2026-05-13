"""Application-layer controller abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.state import State


class FlightController(ABC):
    """Base abstraction for any controller driving the booster."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable controller name."""

    @abstractmethod
    def reset(self, state: State) -> None:
        """Reset any controller-internal state for a fresh episode."""

    @abstractmethod
    def compute_action(self, state: State, dt: float) -> Action:
        """Return the next control action for the current simulation state."""
