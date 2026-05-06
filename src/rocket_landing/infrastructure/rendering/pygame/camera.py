"""Adaptive camera logic for the pygame scene renderer.

This module is intentionally independent from the physics engine.  It only
translates world-space positions expressed in meters into screen-space pixel
coordinates, while maintaining a readable framing for the current descent.
"""

from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


@dataclass(slots=True)
class SceneCamera:
    """Adaptive camera used to keep the rocket readable and cinematic.

    The camera tracks two quantities:
    - ``focus_x``: the world-space lateral point currently centered on screen.
    - ``pixels_per_meter``: the zoom factor used to convert meters into pixels.

    The vertical origin is implicit: the landing pad always stays anchored to a
    fixed screen-space ground line, while altitude is rendered upward from that
    ground line.
    """

    params: RocketParams
    viewport: Viewport
    focus_x: float = 0.0
    pixels_per_meter: float = 3.2

    def update(self, state: State, dt: float) -> None:
        """Smoothly update framing and zoom around the current booster state.

        Args:
            state: Current simulated booster state in world coordinates.
            dt: Frame time used only for smoothing, not for physics.
        """

        desired_x = self._desired_focus_x(state)
        smoothing = min(1.0, dt * 3.5)
        self.focus_x += (desired_x - self.focus_x) * smoothing

        desired_scale = self._desired_pixels_per_meter(state)
        scale_smoothing = min(1.0, dt * 2.8)
        self.pixels_per_meter += (desired_scale - self.pixels_per_meter) * scale_smoothing

    @property
    def pad_screen_x(self) -> float:
        """Return the horizontal screen position of the landing pad center."""

        return self.viewport.width * 0.5

    @property
    def ground_screen_y(self) -> float:
        """Return the screen-space vertical position of the ground plane."""

        return self.viewport.height * 0.855

    def world_to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        """Project a world-space point into screen-space pixel coordinates.

        Args:
            point: ``(x, z)`` tuple expressed in meters.

        Returns:
            Integer pixel coordinates suitable for pygame drawing primitives.
        """

        x, z = point
        screen_x = self.pad_screen_x + ((x - self.focus_x) * self.pixels_per_meter)
        screen_y = self.ground_screen_y - (z * self.pixels_per_meter)
        return int(round(screen_x)), int(round(screen_y))

    def _desired_focus_x(self, state: State) -> float:
        """Bias the camera toward lateral drift while keeping the pad readable."""

        lateral_weight = 0.65
        return state.x * lateral_weight

    def _desired_pixels_per_meter(self, state: State) -> float:
        """Choose a zoom level that keeps the full descent envelope on screen.

        The heuristic keeps both the booster body and a safety margin visible
        above the pad, while clamping the zoom into a viewport-defined range.
        """

        visible_height = max(state.z + (self.params.length * 1.4), 80.0)
        top_margin_px = self.viewport.height * 0.16
        available_px = max(220.0, self.ground_screen_y - top_margin_px)
        dynamic_scale = available_px / visible_height
        return min(
            max(dynamic_scale, self.viewport.min_pixels_per_meter),
            self.viewport.max_pixels_per_meter,
        )
