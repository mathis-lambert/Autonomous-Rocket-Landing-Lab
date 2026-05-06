from __future__ import annotations

import pygame

from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


class HeadsUpDisplay:
    def __init__(self, viewport: Viewport) -> None:
        self._viewport = viewport

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        history: SimulationHistory,
        frame_index: int,
        *,
        playback_speed: float,
    ) -> None:
        state = history.states[frame_index]
        action = history.actions[frame_index]
        lines = [
            f"t = {history.times[frame_index]:6.2f} s    speed = {state.speed:6.2f} m/s",
            f"x = {state.x:7.2f} m   z = {state.z:7.2f} m   theta = {state.theta:6.3f} rad",
            (
                f"vx = {state.vx:6.2f} m/s   vz = {state.vz:6.2f} m/s   "
                f"throttle = {action.throttle:4.2f}   gimbal = {action.gimbal:5.3f} rad   "
                f"playback = x{playback_speed:.2f}"
            ),
        ]
        for index, line in enumerate(lines):
            text = font.render(line, True, self._viewport.text)
            surface.blit(
                text,
                (
                    self._viewport.padding,
                    self._viewport.height - self._viewport.hud_height + index * 24,
                ),
            )
