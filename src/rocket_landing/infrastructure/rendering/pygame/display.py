"""Display setup helpers for pygame renderers."""

from __future__ import annotations

import os

import pygame

from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

WINDOW_FLAGS = pygame.RESIZABLE | pygame.DOUBLEBUF


def enable_high_dpi() -> None:
    """Ask SDL to keep high-DPI framebuffers enabled when the platform supports it."""

    os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "0")


def create_display(viewport: Viewport) -> tuple[pygame.Surface, Viewport]:
    """Create a pygame display surface and return its actual render viewport."""

    surface = pygame.display.set_mode((viewport.width, viewport.height), WINDOW_FLAGS)
    width, height = surface.get_size()
    return surface, viewport.resized(width, height)
