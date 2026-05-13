"""Pygame scene renderer for replay and live simulation frames.

This renderer consumes only immutable history samples plus camera state.  It
does not own any simulation logic and should stay a pure visualization layer.
"""

from __future__ import annotations

import pygame

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.infrastructure.rendering.pygame.assets import SpriteBundle
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.scene_layers import (
    AtmosphereRenderer,
    GroundRenderer,
    RocketRenderer,
)
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

MIN_VELOCITY_VECTOR_SPEED = 0.5
VELOCITY_VECTOR_SECONDS = 0.45
MAX_VELOCITY_VECTOR_LENGTH_PX = 120.0


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
        self._atmosphere = AtmosphereRenderer(viewport, assets, camera)
        self._ground = GroundRenderer(viewport, assets, camera)
        self._rocket = RocketRenderer(params, assets, camera)

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

        self._atmosphere.draw(surface, state)
        self._ground.draw(surface)
        self._draw_trajectory(surface, history, frame_index)
        self._draw_velocity_vector(surface, state)
        self._rocket.draw(surface, state, history.actions[frame_index].throttle)

    def zoom_in(self) -> None:
        """Zoom the scene camera in."""

        self._camera.zoom_in()

    def zoom_out(self) -> None:
        """Zoom the scene camera out."""

        self._camera.zoom_out()

    def reset_zoom(self) -> None:
        """Reset the scene camera zoom."""

        self._camera.reset_zoom()

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
        if magnitude < MIN_VELOCITY_VECTOR_SPEED:
            return

        start = self._camera.world_to_screen((state.x, state.z))
        end_world = (
            state.x + (state.vx * VELOCITY_VECTOR_SECONDS),
            state.z + (state.vz * VELOCITY_VECTOR_SECONDS),
        )
        raw_end = self._camera.world_to_screen(end_world)
        end = self._clamped_screen_vector(start, raw_end, MAX_VELOCITY_VECTOR_LENGTH_PX)
        pygame.draw.line(surface, self._viewport.velocity_vector, start, end, 3)
        pygame.draw.circle(surface, self._viewport.velocity_vector, end, 4)

    @staticmethod
    def _clamped_screen_vector(
        start: tuple[int, int],
        end: tuple[int, int],
        max_length: float,
    ) -> tuple[int, int]:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = ((dx * dx) + (dy * dy)) ** 0.5
        if length <= max_length or length <= 0.0:
            return end

        scale = max_length / length
        return (
            int(round(start[0] + (dx * scale))),
            int(round(start[1] + (dy * scale))),
        )
