"""Pygame scene renderer for replay and live simulation frames.

This renderer consumes only immutable history samples plus camera state.  It
does not own any simulation logic and should stay a pure visualization layer.
"""

from __future__ import annotations

import math

import pygame

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.infrastructure.rendering.pygame.assets import SpriteBundle
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


class PygameReplayScene:
    """Draw the world background, guides, trajectory and booster sprite.

    The same scene object is reused both for replay windows and live sessions.
    """

    def __init__(
        self,
        params: RocketParams,
        viewport: Viewport,
        assets: SpriteBundle,
        camera: SceneCamera,
    ) -> None:
        self._params = params
        self._viewport = viewport
        self._assets = assets
        self._camera = camera
        self._cached_background: pygame.Surface | None = None
        self._cached_background_size: tuple[int, int] | None = None

    def draw(
        self,
        surface: pygame.Surface,
        history: SimulationHistory,
        frame_index: int,
        *,
        dt: float = 0.0,
    ) -> None:
        """Render one frame from the recorded simulation history.

        Args:
            surface: Target pygame surface.
            history: Recorded time-series buffers for the scenario.
            frame_index: Index of the state/action sample to display.
            dt: Presentation delta time used only for camera smoothing.
        """

        state = history.states[frame_index]
        self._camera.update(state, max(dt, 1 / 120))

        self._draw_background(surface)
        self._draw_pad_guides(surface)
        self._draw_trajectory(surface, history, frame_index)
        self._draw_velocity_vector(surface, state)
        self._draw_rocket(surface, state, history.actions[frame_index].throttle)

    def _draw_background(self, surface: pygame.Surface) -> None:
        """Draw and cache the scaled background image for the current viewport."""

        target_size = (self._viewport.width, self._viewport.height)
        if self._cached_background is None or self._cached_background_size != target_size:
            self._cached_background = pygame.transform.smoothscale(
                self._assets.background,
                target_size,
            )
            self._cached_background_size = target_size
        surface.blit(self._cached_background, (0, 0))

        shade = pygame.Surface(target_size, pygame.SRCALPHA)
        shade.fill((0, 18, 42, 26))
        surface.blit(shade, (0, 0))

    def _draw_pad_guides(self, surface: pygame.Surface) -> None:
        """Draw the landing target and centerline guidance markers."""

        center_x = int(round(self._camera.pad_screen_x))
        ground_y = int(round(self._camera.ground_screen_y))
        ring_radius = int(round(24 * self._camera.pixels_per_meter))
        ring_radius = max(36, min(92, ring_radius))

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

        horizon_y = int(self._viewport.height * 0.52)
        pygame.draw.line(
            surface,
            (*self._viewport.guide, 80),
            (center_x, horizon_y),
            (center_x, ground_y),
            1,
        )

    def _draw_trajectory(
        self,
        surface: pygame.Surface,
        history: SimulationHistory,
        frame_index: int,
    ) -> None:
        """Draw the recent portion of the flown trajectory."""

        if frame_index < 2:
            return

        visible_states = history.states[max(0, frame_index - 180) : frame_index + 1]
        points = [self._camera.world_to_screen((state.x, state.z)) for state in visible_states]
        if len(points) >= 2:
            pygame.draw.aalines(surface, self._viewport.trajectory, False, points)

    def _draw_velocity_vector(self, surface: pygame.Surface, state: State) -> None:
        """Draw a short velocity vector for quick motion readability."""

        magnitude = state.speed
        if magnitude < 0.5:
            return

        start = self._camera.world_to_screen((state.x, state.z))
        scale = 0.45
        end_world = (state.x + state.vx * scale, state.z + state.vz * scale)
        end = self._camera.world_to_screen(end_world)
        pygame.draw.line(surface, self._viewport.velocity_vector, start, end, 3)
        pygame.draw.circle(surface, self._viewport.velocity_vector, end, 4)

    def _draw_rocket(self, surface: pygame.Surface, state: State, throttle: float) -> None:
        """Draw the rocket sprite, ground shadow and engine glow."""

        sprite = self._assets.rocket_fire if throttle > 0.03 else self._assets.rocket_off
        sprite_surface = self._transform_rocket_sprite(sprite, state.theta)
        rocket_center = self._camera.world_to_screen((state.x, state.z))
        shadow_center = (rocket_center[0], int(self._camera.ground_screen_y + 8))
        shadow_radius_x = max(22, int(65 / (1.0 + (state.z * 0.02))))
        shadow_radius_y = max(8, int(shadow_radius_x * 0.33))

        shadow = pygame.Surface((shadow_radius_x * 2, shadow_radius_y * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 70), shadow.get_rect())
        shadow_rect = shadow.get_rect(center=shadow_center)
        surface.blit(shadow, shadow_rect)

        rect = sprite_surface.get_rect(center=rocket_center)
        surface.blit(sprite_surface, rect)

        if throttle > 0.05:
            glow_radius = max(12, int(32 * throttle))
            glow = pygame.Surface((glow_radius * 4, glow_radius * 4), pygame.SRCALPHA)
            pygame.draw.circle(
                glow,
                (255, 235, 140, 50),
                (glow.get_width() // 2, glow.get_height() // 2),
                glow_radius,
            )
            glow_rect = glow.get_rect(midtop=(rocket_center[0], rect.bottom - 26))
            surface.blit(glow, glow_rect, special_flags=pygame.BLEND_RGBA_ADD)

    def _transform_rocket_sprite(self, sprite: pygame.Surface, theta: float) -> pygame.Surface:
        """Scale and rotate the rocket sprite to match the world state."""

        body_height_target = self._params.length * self._camera.pixels_per_meter
        scale = body_height_target / self._assets.rocket_body_height_px
        target_size = (
            max(1, int(round(sprite.get_width() * scale))),
            max(1, int(round(sprite.get_height() * scale))),
        )
        scaled = pygame.transform.smoothscale(sprite, target_size)
        # World convention uses positive theta = top of the booster leaning right.
        # Pygame's positive sprite rotation is the opposite of that visual convention here,
        # so we flip the sign to keep rendering aligned with the physics model.
        return pygame.transform.rotozoom(scaled, -math.degrees(theta), 1.0)
