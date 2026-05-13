"""Procedural atmospheric background for the pygame scene."""

from __future__ import annotations

import pygame

from rocket_landing.domain.models.state import State
from rocket_landing.infrastructure.rendering.pygame.assets import SpriteBundle
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

SPACE_START_ALTITUDE_M = 35_000.0
SPACE_FULL_ALTITUDE_M = 90_000.0
CLOUD_FADE_START_ALTITUDE_M = 2_500.0
CLOUD_FADE_END_ALTITUDE_M = 18_000.0
LOW_CLOUD_ALTITUDE_M = 2_200.0
HIGH_CLOUD_ALTITUDE_M = 8_000.0
STAR_TILE_ALPHA_MAX = 190
GRADIENT_STEP_PX = 4

GROUND_SKY_TOP = (88, 154, 213)
GROUND_SKY_BOTTOM = (185, 219, 240)
HIGH_SKY_TOP = (8, 21, 48)
HIGH_SKY_BOTTOM = (32, 74, 126)
SPACE_SKY_TOP = (2, 3, 10)
SPACE_SKY_BOTTOM = (5, 10, 22)
HORIZON_HAZE = (210, 232, 248)


class AtmosphereRenderer:
    """Draw a scalable sky, starfield and cloud layers from physical altitude."""

    def __init__(self, viewport: Viewport, assets: SpriteBundle, camera: SceneCamera) -> None:
        self._viewport = viewport
        self._assets = assets
        self._camera = camera

    def draw(self, surface: pygame.Surface, state: State) -> None:
        """Render the atmosphere for the current altitude."""

        space_ratio = self._altitude_ratio(
            state.z,
            start=SPACE_START_ALTITUDE_M,
            end=SPACE_FULL_ALTITUDE_M,
        )
        self._draw_sky_gradient(surface, space_ratio)
        self._draw_starfield(surface, space_ratio)
        self._draw_horizon_haze(surface, space_ratio)
        self._draw_cloud_layer(
            surface,
            self._assets.cloud_layer_low,
            cloud_altitude=LOW_CLOUD_ALTITUDE_M,
            state=state,
        )
        self._draw_cloud_layer(
            surface,
            self._assets.cloud_layer_high,
            cloud_altitude=HIGH_CLOUD_ALTITUDE_M,
            state=state,
        )

    def _draw_sky_gradient(self, surface: pygame.Surface, space_ratio: float) -> None:
        """Draw an altitude-dependent vertical gradient."""

        top_color = self._mix_color(
            self._mix_color(GROUND_SKY_TOP, HIGH_SKY_TOP, min(1.0, space_ratio * 1.5)),
            SPACE_SKY_TOP,
            space_ratio,
        )
        bottom_color = self._mix_color(
            self._mix_color(GROUND_SKY_BOTTOM, HIGH_SKY_BOTTOM, min(1.0, space_ratio * 1.5)),
            SPACE_SKY_BOTTOM,
            space_ratio,
        )
        for y in range(0, self._viewport.height, GRADIENT_STEP_PX):
            vertical_ratio = y / max(1, self._viewport.height - 1)
            color = self._mix_color(top_color, bottom_color, vertical_ratio)
            pygame.draw.rect(
                surface,
                color,
                pygame.Rect(0, y, self._viewport.width, GRADIENT_STEP_PX),
            )

    def _draw_starfield(self, surface: pygame.Surface, space_ratio: float) -> None:
        """Tile the sparse starfield as the sky transitions to space."""

        if space_ratio <= 0.0:
            return

        tile = self._assets.starfield_tile.copy()
        alpha = int(round(STAR_TILE_ALPHA_MAX * space_ratio))
        tile.set_alpha(alpha)
        tile_width = tile.get_width()
        tile_height = tile.get_height()
        start_x = -int((self._camera.focus_x * 0.02) % tile_width)
        start_y = -int((self._camera.ground_screen_y * 0.04) % tile_height)

        for x in range(start_x, self._viewport.width, tile_width):
            for y in range(start_y, self._viewport.height, tile_height):
                surface.blit(tile, (x, y), special_flags=pygame.BLEND_RGB_ADD)

    def _draw_horizon_haze(self, surface: pygame.Surface, space_ratio: float) -> None:
        """Add a procedural atmospheric limb near the ground line."""

        if space_ratio >= 1.0:
            return

        haze_alpha = int(round(95 * (1.0 - space_ratio)))
        ground_y = int(round(self._camera.ground_screen_y))
        height = max(36, int(self._viewport.height * 0.12))
        haze = pygame.Surface((self._viewport.width, height), pygame.SRCALPHA)
        for y in range(height):
            fade = 1.0 - (y / max(1, height - 1))
            color = (*HORIZON_HAZE, int(haze_alpha * fade))
            pygame.draw.line(haze, color, (0, y), (self._viewport.width, y))
        surface.blit(haze, (0, ground_y - height))

    def _draw_cloud_layer(
        self,
        surface: pygame.Surface,
        cloud: pygame.Surface,
        *,
        cloud_altitude: float,
        state: State,
    ) -> None:
        """Draw one parallax cloud layer at a fixed world altitude."""

        cloud_alpha = int(
            round(
                210
                * (
                    1.0
                    - self._altitude_ratio(
                        state.z,
                        start=CLOUD_FADE_START_ALTITUDE_M,
                        end=CLOUD_FADE_END_ALTITUDE_M,
                    )
                )
            )
        )
        if cloud_alpha <= 0:
            return

        _, screen_y = self._camera.world_to_screen((0.0, cloud_altitude))
        if screen_y < -cloud.get_height() or screen_y > self._viewport.height:
            return

        target_width = max(self._viewport.width, int(round(360 * self._camera.pixels_per_meter)))
        scale = target_width / cloud.get_width()
        target_height = max(1, int(round(cloud.get_height() * scale)))
        scaled = pygame.transform.smoothscale(cloud, (target_width, target_height))
        scaled.set_alpha(cloud_alpha)

        parallax_offset = self._camera.focus_x * self._camera.pixels_per_meter * 0.28
        parallax_x = -int(parallax_offset % target_width)
        draw_y = int(round(screen_y - (target_height * 0.5)))
        end_x = self._viewport.width + target_width
        for x in range(parallax_x - target_width, end_x, target_width):
            surface.blit(scaled, (x, draw_y))

    @staticmethod
    def _altitude_ratio(value: float, *, start: float, end: float) -> float:
        if end <= start:
            return 1.0
        return max(0.0, min(1.0, (value - start) / (end - start)))

    @staticmethod
    def _mix_color(
        start: tuple[int, int, int],
        end: tuple[int, int, int],
        ratio: float,
    ) -> tuple[int, int, int]:
        bounded_ratio = max(0.0, min(1.0, ratio))
        return (
            int(round(start[0] + ((end[0] - start[0]) * bounded_ratio))),
            int(round(start[1] + ((end[1] - start[1]) * bounded_ratio))),
            int(round(start[2] + ((end[2] - start[2]) * bounded_ratio))),
        )
