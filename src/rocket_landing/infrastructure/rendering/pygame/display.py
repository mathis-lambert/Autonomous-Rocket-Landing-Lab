"""Display setup helpers for pygame renderers."""

from __future__ import annotations

import os

import pygame

from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

WINDOW_FLAGS = pygame.RESIZABLE | pygame.DOUBLEBUF
FULLSCREEN_FLAGS = pygame.FULLSCREEN | pygame.DOUBLEBUF


def enable_high_dpi() -> None:
    """Ask SDL to keep high-DPI framebuffers enabled when the platform supports it."""

    os.environ.setdefault("SDL_VIDEO_HIGHDPI_DISABLED", "0")


def desktop_viewport(viewport: Viewport) -> Viewport:
    """Return a viewport sized to the current desktop when pygame exposes it."""

    display_info = pygame.display.Info()
    width = max(viewport.width, display_info.current_w)
    height = max(viewport.height, display_info.current_h)
    return viewport.resized(width, height)


def native_fullscreen_viewport(viewport: Viewport) -> Viewport:
    """Return the highest fullscreen mode SDL reports for the current display."""

    modes = pygame.display.list_modes()
    if not modes or modes == -1:
        return desktop_viewport(viewport)

    width, height = modes[0]
    return viewport.resized(width, height)


def create_display(viewport: Viewport, *, fullscreen: bool) -> tuple[pygame.Surface, Viewport]:
    """Create a pygame display surface and return its actual render viewport."""

    if fullscreen:
        fullscreen_viewport = native_fullscreen_viewport(viewport)
        surface = pygame.display.set_mode(
            (fullscreen_viewport.width, fullscreen_viewport.height),
            FULLSCREEN_FLAGS,
        )
    else:
        surface = pygame.display.set_mode((viewport.width, viewport.height), WINDOW_FLAGS)

    width, height = surface.get_size()
    return surface, viewport.resized(width, height)
