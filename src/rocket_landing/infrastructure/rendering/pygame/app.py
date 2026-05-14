"""Replay-oriented pygame application for inspecting scripted trajectories.

Unlike the live application, this client only consumes an already recorded
history and never mutates the simulation state.
"""

from __future__ import annotations

import pygame

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.infrastructure.rendering.pygame.assets import SpriteAssetLoader, SpriteBundle
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.display import (
    create_display,
    desktop_viewport,
    enable_high_dpi,
)
from rocket_landing.infrastructure.rendering.pygame.hud import HeadsUpDisplay, HudFonts
from rocket_landing.infrastructure.rendering.pygame.scene import PygameReplayScene
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


class PygameReplayApp:
    """Interactive replay client separated from the simulation engine itself.

    It provides lightweight transport controls such as reset and resize, making
    deterministic scenarios easier to inspect visually.
    """

    def __init__(self, params: RocketParams, viewport: Viewport | None = None) -> None:
        self._params = params
        self._viewport = viewport or Viewport()
        self._asset_loader = SpriteAssetLoader()
        self._fullscreen = False
        self._show_force_vectors = False
        self._force_vector_scale_px_per_kn = 0.065
        self._screen: pygame.Surface | None = None
        self._assets: SpriteBundle | None = None
        self._scene: PygameReplayScene | None = None
        self._hud: HeadsUpDisplay | None = None
        self._fonts: HudFonts | None = None

    def run(
        self,
        history: SimulationHistory,
        *,
        title: str = "Rocket landing replay",
        playback_speed: float = 1.0,
        show_force_vectors: bool = False,
        force_vector_scale_px_per_kn: float = 0.065,
    ) -> None:
        """Open a replay window and play a recorded trajectory back in real time."""

        if history.is_empty():
            raise ValueError("history must contain at least one state")
        if playback_speed <= 0.0:
            raise ValueError("playback_speed must be strictly positive")

        self._show_force_vectors = show_force_vectors
        self._force_vector_scale_px_per_kn = force_vector_scale_px_per_kn
        enable_high_dpi()
        pygame.init()
        pygame.display.set_caption(title)
        self._viewport = desktop_viewport(self._viewport)
        self._screen, self._viewport = create_display(self._viewport, fullscreen=self._fullscreen)
        self._assets = self._asset_loader.load()
        self._rebuild_runtime()
        clock = pygame.time.Clock()

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
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_F11:
                    self._toggle_fullscreen()
                elif (
                    event.type == pygame.KEYDOWN
                    and event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS)
                    and self._scene is not None
                ):
                    self._scene.zoom_in()
                elif (
                    event.type == pygame.KEYDOWN
                    and event.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS)
                    and self._scene is not None
                ):
                    self._scene.zoom_out()
                elif (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_0
                    and self._scene is not None
                ):
                    self._scene.reset_zoom()
                elif (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_F3
                    and self._scene is not None
                ):
                    self._scene.toggle_force_vectors()
                elif event.type == pygame.VIDEORESIZE:
                    self._handle_resize(event.w, event.h)

            if not finished and step_duration > 0.0:
                while accumulator >= step_duration and frame_index < len(history.states) - 1:
                    frame_index += 1
                    accumulator -= step_duration
                if frame_index >= len(history.states) - 1:
                    finished = True

            screen, scene, hud, fonts = self._require_runtime()
            state = history.states[frame_index]
            action = history.actions[frame_index]
            scene.draw(screen, history, frame_index, dt=dt_seconds)
            hud.draw(
                screen,
                fonts,
                mode="replay",
                controller_name="constant action",
                state=state,
                action=action,
                elapsed_time=history.times[frame_index],
                status_text="REPLAY" if not finished else "DONE",
                steps_label=f"{frame_index}/{len(history.states) - 1}",
                paused=False,
                debug_forces=scene.show_force_vectors,
            )
            pygame.display.flip()

        pygame.quit()

    def _handle_resize(self, width: int, height: int) -> None:
        """Resize the window and rebuild viewport-dependent renderer objects."""

        if self._fullscreen:
            return

        self._viewport = self._viewport.resized(width, height)
        self._screen, self._viewport = create_display(self._viewport, fullscreen=self._fullscreen)
        self._rebuild_runtime()

    def _toggle_fullscreen(self) -> None:
        """Toggle between native desktop fullscreen and a high-resolution window."""

        self._fullscreen = not self._fullscreen
        if not self._fullscreen:
            self._viewport = desktop_viewport(self._viewport)
        self._screen, self._viewport = create_display(self._viewport, fullscreen=self._fullscreen)
        self._rebuild_runtime()

    def _rebuild_runtime(self) -> None:
        """Recreate scene and HUD objects that depend on viewport dimensions."""

        if self._assets is None:
            raise RuntimeError("sprite assets must be loaded before building the runtime")

        self._scene, self._hud, self._fonts = self._create_runtime(self._assets)

    def _require_runtime(
        self,
    ) -> tuple[pygame.Surface, PygameReplayScene, HeadsUpDisplay, HudFonts]:
        """Return fully initialized pygame runtime objects."""

        if self._screen is None or self._scene is None or self._hud is None or self._fonts is None:
            raise RuntimeError("pygame runtime is not initialized")
        return self._screen, self._scene, self._hud, self._fonts

    def _create_runtime(
        self,
        assets: SpriteBundle,
    ) -> tuple[PygameReplayScene, HeadsUpDisplay, HudFonts]:
        """Instantiate the viewport-dependent rendering helpers."""

        camera = SceneCamera(self._params, self._viewport)
        scene = PygameReplayScene(
            self._params,
            self._viewport,
            assets,
            camera,
            show_force_vectors=self._show_force_vectors,
            force_vector_scale_px_per_kn=self._force_vector_scale_px_per_kn,
        )
        hud = HeadsUpDisplay(self._params, self._viewport)
        fonts = HudFonts.create()
        return scene, hud, fonts
