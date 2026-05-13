"""Sprite loading and preprocessing for the pygame renderers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pygame


@dataclass(frozen=True, slots=True)
class SpriteBundle:
    """In-memory surfaces and derived metadata needed by the scene renderer."""

    rocket_off: pygame.Surface
    rocket_flame_low: pygame.Surface
    rocket_flame_mid: pygame.Surface
    rocket_flame_high: pygame.Surface
    rocket_far_icon: pygame.Surface
    rocket_body_height_px: int
    rocket_anchor_px: tuple[int, int]
    landing_pad: pygame.Surface
    ground_equipment: pygame.Surface
    cloud_layer_low: pygame.Surface
    cloud_layer_high: pygame.Surface
    starfield_tile: pygame.Surface


class SpriteAssetLoader:
    """Loads and preprocesses the 2D assets used by the pygame renderer."""

    def __init__(self) -> None:
        self._sprite_dir = Path(__file__).resolve().parents[5] / "static" / "sprites" / "2d"
        self._bundle: SpriteBundle | None = None

    def load(self) -> SpriteBundle:
        """Load, preprocess and cache the sprite bundle."""

        if self._bundle is not None:
            return self._bundle

        rocket_off = self._load_png("booster_side_off.png")
        body_bounds = self._opaque_bounds(rocket_off)
        body_height = max(1, body_bounds.height)
        rocket_anchor = body_bounds.centerx, body_bounds.bottom
        self._bundle = SpriteBundle(
            rocket_off=rocket_off,
            rocket_flame_low=self._load_png("booster_side_flame_low.png"),
            rocket_flame_mid=self._load_png("booster_side_flame_mid.png"),
            rocket_flame_high=self._load_png("booster_side_flame_high.png"),
            rocket_far_icon=self._load_png("booster_far_icon.png"),
            rocket_body_height_px=body_height,
            rocket_anchor_px=rocket_anchor,
            landing_pad=self._crop_to_opaque(self._load_png("landing_pad_side.png")),
            ground_equipment=self._crop_to_opaque(self._load_png("ground_equipment_silhouette.png")),
            cloud_layer_low=self._crop_to_opaque(self._load_png("cloud_layer_soft_01.png")),
            cloud_layer_high=self._crop_to_opaque(self._load_png("cloud_layer_soft_02.png")),
            starfield_tile=self._load_png("starfield_sparse_tile.png"),
        )
        return self._bundle

    def _load_png(self, filename: str) -> pygame.Surface:
        """Load a PNG asset as an alpha-enabled pygame surface."""

        path = self._sprite_dir / filename
        return pygame.image.load(path.as_posix()).convert_alpha()

    def _crop_to_opaque(self, surface: pygame.Surface) -> pygame.Surface:
        """Trim transparent padding from generated scene props."""

        bounds = self._opaque_bounds(surface)
        return surface.subsurface(bounds).copy()

    def _opaque_bounds(self, surface: pygame.Surface) -> pygame.Rect:
        """Measure the useful opaque sprite bounds for world-space scaling."""

        mask = pygame.mask.from_surface(surface)
        bounding_rects = mask.get_bounding_rects()
        if not bounding_rects:
            return surface.get_rect()
        return max(bounding_rects, key=lambda rect: rect.height)
