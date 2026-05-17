"""Backward-compatible import shim for the keyboard controller."""

from rocket_landing.infrastructure.rendering.pygame.controllers.manual import (
    PygameKeyboardManualController,
)

__all__ = ["PygameKeyboardManualController"]
