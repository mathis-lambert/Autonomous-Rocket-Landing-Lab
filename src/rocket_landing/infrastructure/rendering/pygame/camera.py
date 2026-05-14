"""Adaptive camera logic for the pygame scene renderer.

This module is intentionally independent from the physics engine.  It only
translates world-space positions expressed in meters into screen-space pixel
coordinates, while keeping the vehicle readable at different manual zoom levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

DEFAULT_PIXELS_PER_METER = 3.2
MIN_PIXELS_PER_METER = 0.15
MAX_PIXELS_PER_METER = 8.0
ZOOM_STEP = 1.25
FOCUS_SMOOTHING_RATE = 5.0


@dataclass(slots=True)
class SceneCamera:
    """Adaptive camera used to keep the rocket readable and cinematic.

    The camera tracks three quantities:
    - ``focus_x``: world-space lateral coordinate centered on screen.
    - ``focus_z``: world-space altitude centered on screen.
    - ``pixels_per_meter``: manual zoom factor used to convert meters into pixels.
    """

    params: RocketParams
    viewport: Viewport
    focus_x: float = 0.0
    focus_z: float = 0.0
    pixels_per_meter: float = DEFAULT_PIXELS_PER_METER
    _initialized: bool = field(default=False, init=False, repr=False)

    def update(self, state: State, dt: float) -> None:
        """Smoothly center the camera on the current booster.

        Args:
            state: Current simulated booster state in world coordinates.
            dt: Frame time used only for camera smoothing, not for physics.
        """

        desired_x = state.x
        desired_z = state.z + (self.params.length * 0.5)

        if not self._initialized:
            self.focus_x = desired_x
            self.focus_z = desired_z
            self._initialized = True
            return

        focus_smoothing = min(1.0, dt * FOCUS_SMOOTHING_RATE)
        self.focus_x += (desired_x - self.focus_x) * focus_smoothing
        self.focus_z += (desired_z - self.focus_z) * focus_smoothing

    def zoom_in(self) -> None:
        """Increase camera zoom while preserving world-space units."""

        self.pixels_per_meter = min(MAX_PIXELS_PER_METER, self.pixels_per_meter * ZOOM_STEP)

    def zoom_out(self) -> None:
        """Decrease camera zoom while preserving world-space units."""

        self.pixels_per_meter = max(MIN_PIXELS_PER_METER, self.pixels_per_meter / ZOOM_STEP)

    def reset_zoom(self) -> None:
        """Restore the default readable vehicle scale."""

        self.pixels_per_meter = DEFAULT_PIXELS_PER_METER

    @property
    def pad_screen_x(self) -> float:
        """Return the horizontal screen position of the landing pad center."""

        return (self.viewport.width * 0.5) + (
            (self.params.target_x - self.focus_x) * self.pixels_per_meter
        )

    @property
    def ground_screen_y(self) -> float:
        """Return the screen-space vertical position of the ground plane."""

        return (self.viewport.height * 0.5) + (self.focus_z * self.pixels_per_meter)

    def world_to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        """Project a world-space point into screen-space pixel coordinates.

        Args:
            point: ``(x, z)`` tuple expressed in meters.

        Returns:
            Integer pixel coordinates suitable for pygame drawing primitives.
        """

        x, z = point
        screen_x = (self.viewport.width * 0.5) + ((x - self.focus_x) * self.pixels_per_meter)
        screen_y = (self.viewport.height * 0.5) - ((z - self.focus_z) * self.pixels_per_meter)
        return int(round(screen_x)), int(round(screen_y))
