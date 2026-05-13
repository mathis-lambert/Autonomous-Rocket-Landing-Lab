"""Rocket sprite rendering for the pygame scene."""

from __future__ import annotations

import math

import pygame

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.infrastructure.rendering.pygame.assets import SpriteBundle
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera

FAR_ICON_SCALE_THRESHOLD = 1.45
FAR_ICON_MIN_BODY_HEIGHT_PX = 18
LOW_FLAME_THRESHOLD = 0.03
MID_FLAME_THRESHOLD = 0.45
HIGH_FLAME_THRESHOLD = 0.75
SHADOW_MIN_RADIUS_X = 18
SHADOW_BASE_RADIUS_X = 65
SHADOW_HEIGHT_RATIO = 0.33
SHADOW_ALTITUDE_FALLOFF = 0.02
SHADOW_ALPHA = 70
SHADOW_GROUND_OFFSET_PX = 8


class RocketRenderer:
    """Draw the vehicle with throttle-dependent sprite selection."""

    def __init__(self, params: RocketParams, assets: SpriteBundle, camera: SceneCamera) -> None:
        self._params = params
        self._assets = assets
        self._camera = camera

    def draw(self, surface: pygame.Surface, state: State, throttle: float) -> None:
        """Draw the rocket and its ground shadow."""

        use_far_icon = self._camera.pixels_per_meter < FAR_ICON_SCALE_THRESHOLD
        sprite = self._select_sprite(throttle, use_far_icon=use_far_icon)
        min_body_height = FAR_ICON_MIN_BODY_HEIGHT_PX if use_far_icon else 1
        sprite_surface, sprite_scale = self._transform_sprite(
            sprite,
            state.theta,
            min_body_height=min_body_height,
        )
        rocket_anchor = self._camera.world_to_screen((state.x, state.z))

        self._draw_ground_shadow(surface, state, rocket_anchor)
        rect = self._anchored_rect(sprite_surface, sprite, state.theta, rocket_anchor, sprite_scale)
        surface.blit(sprite_surface, rect)

    def _select_sprite(self, throttle: float, *, use_far_icon: bool) -> pygame.Surface:
        if use_far_icon:
            return self._assets.rocket_far_icon
        if throttle < LOW_FLAME_THRESHOLD:
            return self._assets.rocket_off
        if throttle < MID_FLAME_THRESHOLD:
            return self._assets.rocket_flame_low
        if throttle < HIGH_FLAME_THRESHOLD:
            return self._assets.rocket_flame_mid
        return self._assets.rocket_flame_high

    def _draw_ground_shadow(
        self,
        surface: pygame.Surface,
        state: State,
        rocket_center: tuple[int, int],
    ) -> None:
        shadow_center = (
            rocket_center[0],
            int(self._camera.ground_screen_y + SHADOW_GROUND_OFFSET_PX),
        )
        shadow_radius_x = max(
            SHADOW_MIN_RADIUS_X,
            int(SHADOW_BASE_RADIUS_X / (1.0 + (state.z * SHADOW_ALTITUDE_FALLOFF))),
        )
        shadow_radius_y = max(1, int(shadow_radius_x * SHADOW_HEIGHT_RATIO))

        shadow = pygame.Surface((shadow_radius_x * 2, shadow_radius_y * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, SHADOW_ALPHA), shadow.get_rect())
        surface.blit(shadow, shadow.get_rect(center=shadow_center))

    def _transform_sprite(
        self,
        sprite: pygame.Surface,
        theta: float,
        *,
        min_body_height: int,
    ) -> tuple[pygame.Surface, float]:
        body_height_target = max(
            min_body_height,
            self._params.length * self._camera.pixels_per_meter,
        )
        scale = body_height_target / self._assets.rocket_body_height_px
        target_size = (
            max(1, int(round(sprite.get_width() * scale))),
            max(1, int(round(sprite.get_height() * scale))),
        )
        scaled = pygame.transform.smoothscale(sprite, target_size)
        return pygame.transform.rotozoom(scaled, -math.degrees(theta), 1.0), scale

    def _anchored_rect(
        self,
        transformed_sprite: pygame.Surface,
        source_sprite: pygame.Surface,
        theta: float,
        screen_anchor: tuple[int, int],
        sprite_scale: float,
    ) -> pygame.Rect:
        """Place the transformed sprite so the booster base matches the world anchor."""

        anchor_x, anchor_y = self._assets.rocket_anchor_px
        source_center_x = source_sprite.get_width() * 0.5
        source_center_y = source_sprite.get_height() * 0.5
        offset_x = (anchor_x - source_center_x) * sprite_scale
        offset_y = (anchor_y - source_center_y) * sprite_scale
        rotated_offset_x, rotated_offset_y = self._rotate_screen_offset(offset_x, offset_y, -theta)
        center = (
            int(round(screen_anchor[0] - rotated_offset_x)),
            int(round(screen_anchor[1] - rotated_offset_y)),
        )
        return transformed_sprite.get_rect(center=center)

    @staticmethod
    def _rotate_screen_offset(
        offset_x: float,
        offset_y: float,
        angle: float,
    ) -> tuple[float, float]:
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        return (
            (offset_x * cos_angle) - (offset_y * sin_angle),
            (offset_x * sin_angle) + (offset_y * cos_angle),
        )
