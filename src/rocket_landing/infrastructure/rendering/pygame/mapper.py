from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


@dataclass(frozen=True, slots=True)
class WorldBounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float

    @classmethod
    def from_history(cls, history: SimulationHistory, params: RocketParams) -> WorldBounds:
        xs = [state.x for state in history.states]
        zs = [state.z for state in history.states]
        horizontal_padding = max(params.length, 20.0)
        vertical_padding = max(params.length, 20.0)
        return cls(
            min_x=min(xs) - horizontal_padding,
            max_x=max(xs) + horizontal_padding,
            min_z=min(-5.0, min(zs) - params.length * 0.25),
            max_z=max(zs) + vertical_padding,
        )


class ScreenMapper:
    def __init__(self, bounds: WorldBounds, viewport: Viewport) -> None:
        self._bounds = bounds
        self._viewport = viewport
        self._pixels_per_meter = min(viewport.pixels_per_meter, self._fit_pixels_per_meter())

    @property
    def viewport(self) -> Viewport:
        return self._viewport

    @property
    def bounds(self) -> WorldBounds:
        return self._bounds

    def to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        x, z = point
        screen_x = self._viewport.padding + (x - self._bounds.min_x) * self._pixels_per_meter
        screen_y = self._viewport.padding + (self._bounds.max_z - z) * self._pixels_per_meter
        return int(round(screen_x)), int(round(screen_y))

    def _fit_pixels_per_meter(self) -> float:
        world_width = max(1.0, self._bounds.max_x - self._bounds.min_x)
        world_height = max(1.0, self._bounds.max_z - self._bounds.min_z)
        usable_width = self._viewport.width - 2 * self._viewport.padding
        usable_height = (
            self._viewport.height - self._viewport.hud_height - 2 * self._viewport.padding
        )
        return min(usable_width / world_width, usable_height / world_height)
