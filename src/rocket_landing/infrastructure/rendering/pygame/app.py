from __future__ import annotations

import pygame

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.infrastructure.rendering.pygame.hud import HeadsUpDisplay
from rocket_landing.infrastructure.rendering.pygame.mapper import ScreenMapper, WorldBounds
from rocket_landing.infrastructure.rendering.pygame.scene import PygameReplayScene
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


class PygameReplayApp:
    """Interactive replay client separated from the simulation engine itself."""

    def __init__(self, params: RocketParams, viewport: Viewport | None = None) -> None:
        self._params = params
        self._viewport = viewport or Viewport()

    def run(
        self,
        history: SimulationHistory,
        *,
        title: str = "Rocket landing replay",
        playback_speed: float = 1.0,
    ) -> None:
        if history.is_empty():
            raise ValueError("history must contain at least one state")
        if playback_speed <= 0.0:
            raise ValueError("playback_speed must be strictly positive")

        pygame.init()
        pygame.display.set_caption(title)

        mapper = ScreenMapper(WorldBounds.from_history(history, self._params), self._viewport)
        scene = PygameReplayScene(self._params, mapper)
        hud = HeadsUpDisplay(self._viewport)

        screen = pygame.display.set_mode((self._viewport.width, self._viewport.height))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 20)

        frame_index = 0
        finished = False
        accumulator = 0.0
        step_duration = history.times[1] - history.times[0] if len(history.times) > 1 else 0.02
        running = True

        while running:
            dt_seconds = clock.tick(60) / 1000.0
            accumulator += dt_seconds * playback_speed

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    frame_index = 0
                    accumulator = 0.0
                    finished = False

            if not finished and step_duration > 0.0:
                while accumulator >= step_duration and frame_index < len(history.states) - 1:
                    frame_index += 1
                    accumulator -= step_duration
                if frame_index >= len(history.states) - 1:
                    finished = True

            scene.draw(screen, history, frame_index)
            hud.draw(screen, font, history, frame_index, playback_speed=playback_speed)
            pygame.display.flip()

        pygame.quit()
