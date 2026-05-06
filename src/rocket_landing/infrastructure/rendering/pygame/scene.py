from __future__ import annotations

import math

import pygame

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.geometry import BoosterGeometry
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.infrastructure.rendering.pygame.mapper import ScreenMapper


class PygameReplayScene:
    def __init__(self, params: RocketParams, mapper: ScreenMapper) -> None:
        self._mapper = mapper
        self._geometry = BoosterGeometry(params)

    def draw(
        self,
        surface: pygame.Surface,
        history: SimulationHistory,
        frame_index: int,
    ) -> None:
        self._draw_background(surface)
        self._draw_trajectory(surface, history, frame_index)
        self._draw_booster(
            surface,
            history.states[frame_index],
            history.actions[frame_index].throttle,
        )

    def _draw_background(self, surface: pygame.Surface) -> None:
        viewport = self._mapper.viewport
        bounds = self._mapper.bounds
        surface.fill(viewport.background)
        ground_y = self._mapper.to_screen((0.0, 0.0))[1]
        pygame.draw.rect(
            surface,
            viewport.ground,
            pygame.Rect(0, ground_y, viewport.width, viewport.height - ground_y),
        )

        spacing_m = 20.0
        x_value = math.floor(bounds.min_x / spacing_m) * spacing_m
        right = math.ceil(bounds.max_x / spacing_m) * spacing_m
        while x_value <= right:
            start = self._mapper.to_screen((x_value, bounds.min_z))
            end = self._mapper.to_screen((x_value, bounds.max_z))
            pygame.draw.line(surface, viewport.grid, start, end, 1)
            x_value += spacing_m

    def _draw_trajectory(
        self,
        surface: pygame.Surface,
        history: SimulationHistory,
        frame_index: int,
    ) -> None:
        if frame_index < 1:
            return

        points = [
            self._mapper.to_screen((state.x, state.z))
            for state in history.states[: frame_index + 1]
        ]
        pygame.draw.lines(surface, self._mapper.viewport.trajectory, False, points, 2)

    def _draw_booster(self, surface: pygame.Surface, state: State, throttle: float) -> None:
        bottom_world, top_world = self._geometry.segment_endpoints(state)
        bottom = self._mapper.to_screen(bottom_world)
        top = self._mapper.to_screen(top_world)
        viewport = self._mapper.viewport

        pygame.draw.line(surface, viewport.booster, bottom, top, 6)
        pygame.draw.circle(surface, viewport.booster, top, 6)

        if throttle <= 0.01:
            return

        body_dx = top[0] - bottom[0]
        body_dy = top[1] - bottom[1]
        body_length = max(1.0, math.hypot(body_dx, body_dy))
        unit_x = body_dx / body_length
        unit_y = body_dy / body_length
        flame_length = max(10.0, 28.0 * throttle)
        flame_tip = (
            int(round(bottom[0] - unit_x * flame_length)),
            int(round(bottom[1] - unit_y * flame_length)),
        )
        flame_half_width = 5
        perp_x = -unit_y
        perp_y = unit_x
        flame_left = (
            int(round(bottom[0] + perp_x * flame_half_width)),
            int(round(bottom[1] + perp_y * flame_half_width)),
        )
        flame_right = (
            int(round(bottom[0] - perp_x * flame_half_width)),
            int(round(bottom[1] - perp_y * flame_half_width)),
        )
        pygame.draw.polygon(surface, viewport.flame, [flame_left, flame_right, flame_tip])
