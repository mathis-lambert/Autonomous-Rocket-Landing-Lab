"""Common pygame simulation app driven by interchangeable controllers."""

from __future__ import annotations

import pygame

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.infrastructure.rendering.pygame.assets import SpriteAssetLoader, SpriteBundle
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.controllers.base import SimulationController
from rocket_landing.infrastructure.rendering.pygame.display import create_display, enable_high_dpi
from rocket_landing.infrastructure.rendering.pygame.hud import HeadsUpDisplay, HudFonts
from rocket_landing.infrastructure.rendering.pygame.runtime import SimulationRuntime
from rocket_landing.infrastructure.rendering.pygame.scene import PygameSimulationScene
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


class PygameSimulationApp:
    """Realtime simulation app for manual or policy-driven runtimes."""

    def __init__(
        self,
        params: RocketParams,
        runtime: SimulationRuntime,
        *,
        controller: SimulationController,
        viewport: Viewport | None = None,
        show_force_vectors: bool = False,
        force_vector_scale_px_per_kn: float = 0.065,
    ) -> None:
        self._params = params
        self._runtime = runtime
        self._controller = controller
        self._viewport = viewport or Viewport()
        self._paused = False
        self._show_force_vectors = show_force_vectors
        self._force_vector_scale_px_per_kn = force_vector_scale_px_per_kn
        self._asset_loader = SpriteAssetLoader()
        self._screen: pygame.Surface | None = None
        self._assets: SpriteBundle | None = None
        self._scene: PygameSimulationScene | None = None
        self._hud: HeadsUpDisplay | None = None
        self._fonts: HudFonts | None = None

    def run(self, *, title: str = "Rocket landing simulation") -> None:
        """Open the live window and drive the simulation at interactive frame rate."""

        enable_high_dpi()
        pygame.init()
        pygame.display.set_caption(title)

        self._screen, self._viewport = create_display(self._viewport)
        self._assets = self._asset_loader.load()
        clock = pygame.time.Clock()
        self._rebuild_runtime()
        self._reset_runtime()

        running = True
        while running:
            frame_dt = clock.tick(60) / 1000.0
            running = self._handle_events()
            if not running:
                break

            if not self._paused and not self._runtime.is_finished:
                self._update_controller(frame_dt)
                self._runtime.step(self._controller.current_action())
                self._sync_controller_with_runtime()

            screen, scene, hud, fonts = self._require_runtime()
            frame_index = self._runtime.current_frame_index
            action = self._runtime.latest_action
            scene.draw(screen, self._runtime.history, frame_index, dt=frame_dt)
            hud.draw(
                screen,
                fonts,
                state=self._runtime.state,
                action=action,
                elapsed_time=self._runtime.time,
                status_text=self._status_text(),
                steps_label=f"{self._runtime.step_count}/{self._runtime.max_steps}",
                paused=self._paused,
                debug_forces=scene.show_force_vectors,
            )
            pygame.display.flip()

        pygame.quit()

    def _handle_events(self) -> bool:
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
                self._reset_runtime()
            elif (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_F3
                and self._scene is not None
            ):
                self._scene.toggle_force_vectors()
        return True

    def _handle_resize(self, width: int, height: int) -> None:
        self._viewport = self._viewport.resized(width, height)
        self._screen, self._viewport = create_display(self._viewport)
        self._rebuild_runtime()

    def _update_controller(self, frame_dt: float) -> None:
        pressed = pygame.key.get_pressed()
        self._controller.handle_pressed_keys(pressed, frame_dt)

    def _reset_runtime(self) -> None:
        self._runtime.reset()
        self._paused = False
        self._controller.reset()
        self._sync_controller_with_runtime()

    def _sync_controller_with_runtime(self) -> None:
        self._controller.sync_from_state(self._runtime.state)

    def _status_text(self) -> str:
        last_result = self._runtime.last_result
        if last_result is None:
            return "READY"
        if last_result.landed:
            return "LANDED"
        if last_result.crashed:
            return "CRASH"
        if self._paused:
            return "PAUSED"
        if (
            abs(self._runtime.state.vz) < 1.5
            and self._runtime.state.ground_clearance(self._params.length) > 5.0
        ):
            return "HOVER"
        return "DESCENT"

    def _rebuild_runtime(self) -> None:
        if self._assets is None:
            raise RuntimeError("sprite assets must be loaded before building the runtime")

        camera = SceneCamera(self._params, self._viewport)
        self._scene = PygameSimulationScene(
            self._params,
            self._viewport,
            self._assets,
            camera,
            show_force_vectors=self._show_force_vectors,
            force_vector_scale_px_per_kn=self._force_vector_scale_px_per_kn,
        )
        self._hud = HeadsUpDisplay(self._params, self._viewport)
        self._fonts = HudFonts.create()

    def _require_runtime(
        self,
    ) -> tuple[pygame.Surface, PygameSimulationScene, HeadsUpDisplay, HudFonts]:
        if self._screen is None or self._scene is None or self._hud is None or self._fonts is None:
            raise RuntimeError("pygame runtime is not initialized")
        return self._screen, self._scene, self._hud, self._fonts
