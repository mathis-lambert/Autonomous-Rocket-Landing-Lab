"""Realtime pygame application that couples controllers to the simulation session.

This module is the outermost interactive shell of the project.  It coordinates
input events, controller selection, session stepping and viewport-dependent
rendering, while keeping those concerns separated from the physics domain.
"""

from __future__ import annotations

import pygame

from rocket_landing.application.control.controller import FlightController
from rocket_landing.application.use_cases.run_controlled_session import ControlledSimulationSession
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.infrastructure.rendering.pygame.assets import SpriteAssetLoader, SpriteBundle
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.hud import HeadsUpDisplay, HudFonts
from rocket_landing.infrastructure.rendering.pygame.manual_controller import (
    PygameKeyboardManualController,
)
from rocket_landing.infrastructure.rendering.pygame.scene import PygameReplayScene
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


class PygameLiveSimulationApp:
    """Realtime simulation app that plugs controllers into the simulation session.

    The application owns the pygame runtime objects but delegates:
    - action computation to controller objects
    - state evolution to the controlled simulation session
    - drawing to the scene and HUD helpers
    """

    def __init__(
        self,
        params: RocketParams,
        session: ControlledSimulationSession,
        *,
        controllers: list[FlightController],
        active_controller_name: str,
        viewport: Viewport | None = None,
    ) -> None:
        if not controllers:
            raise ValueError("controllers must contain at least one controller")

        self._params = params
        self._session = session
        self._controllers = controllers
        self._viewport = viewport or Viewport()
        self._active_controller_index = self._find_controller_index(active_controller_name)
        self._paused = False
        self._asset_loader = SpriteAssetLoader()
        self._screen: pygame.Surface | None = None
        self._assets: SpriteBundle | None = None
        self._scene: PygameReplayScene | None = None
        self._hud: HeadsUpDisplay | None = None
        self._fonts: HudFonts | None = None

    def run(self, *, title: str = "Rocket landing live session") -> None:
        """Open the live window and drive the simulation at interactive frame rate.

        The loop runs at the display frame rate while the simulation itself
        advances with the fixed ``dt`` configured in the session.
        """

        pygame.init()
        pygame.display.set_caption(title)

        self._screen = pygame.display.set_mode(
            (self._viewport.width, self._viewport.height),
            pygame.RESIZABLE,
        )
        self._assets = self._asset_loader.load()
        clock = pygame.time.Clock()
        self._rebuild_runtime()

        self._reset_session()

        running = True
        while running:
            frame_dt = clock.tick(60) / 1000.0
            running = self._handle_events()
            if not running:
                break

            if not self._paused and not self._session.is_finished:
                self._update_manual_input(frame_dt)
                action = self._active_controller.compute_action(
                    self._session.state,
                    self._session.dt,
                )
                self._session.step(action)

            screen, scene, hud, fonts = self._require_runtime()

            frame_index = len(self._session.history.states) - 1
            action = self._session.history.actions[frame_index]
            scene.draw(screen, self._session.history, frame_index, dt=frame_dt)
            hud.draw(
                screen,
                fonts,
                mode="live",
                controller_name=self._active_controller.name,
                state=self._session.state,
                action=action,
                elapsed_time=self._session.time,
                status_text=self._status_text(),
                steps_label=f"{self._session.step_count}/{self._session.max_steps}",
                paused=self._paused,
                extra_lines=self._hud_lines(),
            )
            pygame.display.flip()

        pygame.quit()

    @property
    def _active_controller(self) -> FlightController:
        """Return the controller currently selected by the user."""

        return self._controllers[self._active_controller_index]

    def _handle_events(self) -> bool:
        """Process window and keyboard events for the current frame."""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.VIDEORESIZE:
                self._handle_resize(event.w, event.h)
                continue
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self._paused = not self._paused
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                self._reset_session()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                self._cycle_controller()

            if isinstance(self._active_controller, PygameKeyboardManualController):
                self._active_controller.handle_event(event)
        return True

    def _handle_resize(self, width: int, height: int) -> None:
        """Resize the window and rebuild viewport-dependent renderer state."""

        self._viewport = self._viewport.resized(width, height)
        self._screen = pygame.display.set_mode(
            (self._viewport.width, self._viewport.height),
            pygame.RESIZABLE,
        )
        self._rebuild_runtime()

    def _update_manual_input(self, frame_dt: float) -> None:
        """Poll held keys for the manual controller."""

        if not isinstance(self._active_controller, PygameKeyboardManualController):
            return
        pressed = pygame.key.get_pressed()
        self._active_controller.update_from_pressed_keys(pressed, frame_dt)

    def _reset_session(self) -> None:
        """Reset the simulation session and every registered controller."""

        self._session.reset()
        self._paused = False
        for controller in self._controllers:
            controller.reset(self._session.state)

    def _cycle_controller(self) -> None:
        """Switch to the next available controller implementation."""

        self._active_controller_index = (self._active_controller_index + 1) % len(self._controllers)
        self._active_controller.reset(self._session.state)

    def _find_controller_index(self, name: str) -> int:
        """Resolve a controller name into its position in the controller list."""

        for index, controller in enumerate(self._controllers):
            if controller.name == name:
                return index
        raise ValueError(f"unknown controller: {name}")

    def _hud_lines(self) -> list[str]:
        """Build extra HUD lines shared by all live-session frames."""

        control_line = "TAB switch controller   SPACE pause   R reset   ESC quit"
        return [control_line, *self._active_controller.status_lines()]

    def _status_text(self) -> str:
        """Derive a short human-readable flight status for the HUD."""

        if self._session.last_result is None:
            return "READY"
        if self._session.last_result.landed:
            return "LANDED"
        if self._session.last_result.crashed:
            return "CRASH"
        if self._paused:
            return "PAUSED"
        if abs(self._session.state.vz) < 1.5 and self._session.state.z > 5.0:
            return "HOVER"
        return "DESCENT"

    def _rebuild_runtime(self) -> None:
        """Recreate scene and HUD objects after initialization or resize."""

        if self._assets is None:
            raise RuntimeError("sprite assets must be loaded before building the runtime")

        scene, hud, fonts = self._create_runtime(self._assets)
        self._scene = scene
        self._hud = hud
        self._fonts = fonts

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
        scene = PygameReplayScene(self._params, self._viewport, assets, camera)
        hud = HeadsUpDisplay(self._params, self._viewport)
        fonts = HudFonts.create()
        return scene, hud, fonts
