"""Common controller contract for pygame simulation apps."""

from __future__ import annotations

from typing import Protocol

import pygame

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.state import State


class SimulationController(Protocol):
    """Minimal controller interface used by the pygame simulation app."""

    def reset(self) -> None:
        """Reset any internal controller state."""

    def sync_from_state(self, state: State) -> None:
        """Receive the latest simulator state before the next action is queried."""

    def handle_pressed_keys(
        self,
        pressed_keys: pygame.key.ScancodeWrapper,
        dt: float,
    ) -> None:
        """Update controller state from the currently pressed keyboard state."""

    def current_action(self) -> Action:
        """Return the action that should be applied on this frame."""
