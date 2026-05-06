"""Sprite loading and preprocessing for the pygame renderers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame


@dataclass(frozen=True, slots=True)
class SpriteBundle:
    """In-memory surfaces and derived metadata needed by the scene renderer."""

    background: pygame.Surface
    rocket_off: pygame.Surface
    rocket_fire: pygame.Surface
    rocket_body_height_px: int


class SpriteAssetLoader:
    """Loads and preprocesses the 2D assets used by the pygame renderer."""

    def __init__(self) -> None:
        self._sprite_dir = Path(__file__).resolve().parents[5] / "static" / "sprites" / "2d"
        self._bundle: SpriteBundle | None = None

    def load(self) -> SpriteBundle:
        """Load, preprocess and cache the sprite bundle."""

        if self._bundle is not None:
            return self._bundle

        background = self._load_png("main_background_daylight.png")
        rocket_off = self._prepare_rocket_sprite(self._load_png("falcon_heavy_2d_off.png"))
        rocket_fire = self._prepare_rocket_sprite(self._load_png("falcon_heavy_2d_fire.png"))
        body_height = max(1, self._opaque_bounds_height(rocket_off))
        self._bundle = SpriteBundle(
            background=background,
            rocket_off=rocket_off,
            rocket_fire=rocket_fire,
            rocket_body_height_px=body_height,
        )
        return self._bundle

    def _load_png(self, filename: str) -> pygame.Surface:
        """Load a PNG asset as an alpha-enabled pygame surface."""

        path = self._sprite_dir / filename
        return pygame.image.load(path.as_posix()).convert_alpha()

    def _prepare_rocket_sprite(self, surface: pygame.Surface) -> pygame.Surface:
        """Remove the flat background color from a raw rocket sprite."""

        prepared = surface.copy()
        transparent_color = self._average_corner_color(prepared)
        self._remove_background_color(prepared, transparent_color, tolerance=42)
        return prepared

    def _average_corner_color(self, surface: pygame.Surface) -> tuple[int, int, int]:
        """Estimate the background color by averaging the four corners."""

        pixels = [
            surface.get_at((0, 0)),
            surface.get_at((surface.get_width() - 1, 0)),
            surface.get_at((0, surface.get_height() - 1)),
            surface.get_at((surface.get_width() - 1, surface.get_height() - 1)),
        ]
        red = sum(pixel.r for pixel in pixels) // len(pixels)
        green = sum(pixel.g for pixel in pixels) // len(pixels)
        blue = sum(pixel.b for pixel in pixels) // len(pixels)
        return red, green, blue

    def _remove_background_color(
        self,
        surface: pygame.Surface,
        key_color: tuple[int, int, int],
        *,
        tolerance: int,
    ) -> None:
        """Make pixels close to the sampled background color transparent."""

        width = surface.get_width()
        height = surface.get_height()
        key_r, key_g, key_b = key_color

        surface.lock()
        try:
            for x in range(width):
                for y in range(height):
                    pixel = surface.get_at((x, y))
                    if (
                        abs(pixel.r - key_r) <= tolerance
                        and abs(pixel.g - key_g) <= tolerance
                        and abs(pixel.b - key_b) <= tolerance
                    ):
                        surface.set_at((x, y), pygame.Color(pixel.r, pixel.g, pixel.b, 0))
        finally:
            surface.unlock()

    def _opaque_bounds_height(self, surface: pygame.Surface) -> int:
        """Measure the useful opaque sprite height for world-space scaling."""

        mask = pygame.mask.from_surface(surface)
        bounding_rects = mask.get_bounding_rects()
        if not bounding_rects:
            return surface.get_height()
        return max(rect.height for rect in bounding_rects)
