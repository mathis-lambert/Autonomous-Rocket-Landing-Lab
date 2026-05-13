"""Ground, pad and landing-site props for the pygame scene."""

from __future__ import annotations

import pygame

from rocket_landing.infrastructure.rendering.pygame.assets import SpriteBundle
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

PAD_WORLD_WIDTH_M = 72.0
EQUIPMENT_WORLD_HEIGHT_M = 26.0
EQUIPMENT_WORLD_X_M = 82.0
GROUND_BAND_HEIGHT_PX = 80
PAD_GUIDE_RADIUS_M = 24.0
PAD_GUIDE_MIN_RADIUS_PX = 28
PAD_GUIDE_MAX_RADIUS_PX = 92
PAD_TOP_OFFSET_PX = 2
EQUIPMENT_BOTTOM_OFFSET_PX = 4
GROUND_COLOR = (28, 31, 34)
GROUND_TOP_COLOR = (82, 84, 78)
CENTERLINE_TOP_RATIO = 0.52


class GroundRenderer:
    """Draw the landing pad, simple ground band and scale reference props."""

    def __init__(self, viewport: Viewport, assets: SpriteBundle, camera: SceneCamera) -> None:
        self._viewport = viewport
        self._assets = assets
        self._camera = camera

    def draw(self, surface: pygame.Surface) -> None:
        """Render ground visuals around the landing pad."""

        ground_y = int(round(self._camera.ground_screen_y))
        self._draw_ground_band(surface, ground_y)
        self._draw_landing_pad(surface, ground_y)
        self._draw_ground_equipment(surface, ground_y)
        self._draw_pad_guides(surface, ground_y)

    def _draw_ground_band(self, surface: pygame.Surface, ground_y: int) -> None:
        ground_rect = pygame.Rect(
            0,
            ground_y,
            self._viewport.width,
            max(0, self._viewport.height - ground_y),
        )
        if ground_rect.height <= 0:
            return
        pygame.draw.rect(surface, GROUND_COLOR, ground_rect)
        pygame.draw.line(
            surface,
            GROUND_TOP_COLOR,
            (0, ground_y),
            (self._viewport.width, ground_y),
            2,
        )

    def _draw_landing_pad(self, surface: pygame.Surface, ground_y: int) -> None:
        pad = self._assets.landing_pad
        target_width = max(1, int(round(PAD_WORLD_WIDTH_M * self._camera.pixels_per_meter)))
        target_height = max(1, int(round(pad.get_height() * (target_width / pad.get_width()))))
        scaled = pygame.transform.smoothscale(pad, (target_width, target_height))
        rect = scaled.get_rect(
            midtop=(int(round(self._camera.pad_screen_x)), ground_y - PAD_TOP_OFFSET_PX)
        )
        surface.blit(scaled, rect)

    def _draw_ground_equipment(self, surface: pygame.Surface, ground_y: int) -> None:
        equipment = self._assets.ground_equipment
        target_height = max(1, int(round(EQUIPMENT_WORLD_HEIGHT_M * self._camera.pixels_per_meter)))
        target_width = max(
            1,
            int(round(equipment.get_width() * (target_height / equipment.get_height()))),
        )
        scaled = pygame.transform.smoothscale(equipment, (target_width, target_height))
        x, _ = self._camera.world_to_screen((EQUIPMENT_WORLD_X_M, 0.0))
        rect = scaled.get_rect(midbottom=(x, ground_y + EQUIPMENT_BOTTOM_OFFSET_PX))
        surface.blit(scaled, rect)

    def _draw_pad_guides(self, surface: pygame.Surface, ground_y: int) -> None:
        center_x = int(round(self._camera.pad_screen_x))
        ring_radius = int(round(PAD_GUIDE_RADIUS_M * self._camera.pixels_per_meter))
        ring_radius = max(PAD_GUIDE_MIN_RADIUS_PX, min(PAD_GUIDE_MAX_RADIUS_PX, ring_radius))

        pygame.draw.circle(
            surface,
            self._viewport.target_ring,
            (center_x, ground_y),
            ring_radius,
            3,
        )
        pygame.draw.circle(
            surface,
            self._viewport.target_ring,
            (center_x, ground_y),
            ring_radius // 2,
            2,
        )
        pygame.draw.line(
            surface,
            self._viewport.target_ring,
            (center_x - ring_radius, ground_y),
            (center_x + ring_radius, ground_y),
            2,
        )
        pygame.draw.line(
            surface,
            self._viewport.target_ring,
            (center_x, ground_y - ring_radius),
            (center_x, ground_y + ring_radius),
            2,
        )

        horizon_y = int(self._viewport.height * CENTERLINE_TOP_RATIO)
        pygame.draw.line(
            surface,
            (*self._viewport.guide, 80),
            (center_x, horizon_y),
            (center_x, ground_y),
            1,
        )
