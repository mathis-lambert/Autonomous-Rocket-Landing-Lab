"""Realtime pygame application for watching a learned policy fly the booster."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pygame

from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.infrastructure.rendering.pygame.assets import SpriteAssetLoader, SpriteBundle
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.display import create_display, enable_high_dpi
from rocket_landing.infrastructure.rendering.pygame.hud import HeadsUpDisplay, HudFonts
from rocket_landing.infrastructure.rendering.pygame.scene import PygameSimulationScene
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

PolicyFn = Callable[[np.ndarray], np.ndarray]


class PygamePolicySimulationApp:
    """Drive the simulation using a policy while reusing the existing renderer."""

    def __init__(
        self,
        env: RocketLanderEnv,
        *,
        policy_fn: PolicyFn,
        viewport: Viewport | None = None,
        show_force_vectors: bool = False,
        force_vector_scale_px_per_kn: float = 0.065,
    ) -> None:
        self._env = env
        self._policy_fn = policy_fn
        self._viewport = viewport or Viewport()
        self._paused = False
        self._episode_finished = False
        self._show_force_vectors = show_force_vectors
        self._force_vector_scale_px_per_kn = force_vector_scale_px_per_kn
        self._asset_loader = SpriteAssetLoader()
        self._screen: pygame.Surface | None = None
        self._assets: SpriteBundle | None = None
        self._scene: PygameSimulationScene | None = None
        self._hud: HeadsUpDisplay | None = None
        self._fonts: HudFonts | None = None
        self._observation: np.ndarray | None = None

    def run(self, *, title: str = "Rocket landing policy playback") -> None:
        """Open a pygame window and let the policy fly one episode at a time."""

        enable_high_dpi()
        pygame.init()
        pygame.display.set_caption(title)

        self._screen, self._viewport = create_display(self._viewport)
        self._assets = self._asset_loader.load()
        clock = pygame.time.Clock()
        self._rebuild_runtime()
        self._reset_episode()

        running = True
        while running:
            frame_dt = clock.tick(60) / 1000.0
            running = self._handle_events()
            if not running:
                break

            if not self._paused and not self._episode_finished:
                self._step_policy()

            screen, scene, hud, fonts = self._require_runtime()
            session = self._env.session
            frame_index = session.current_frame_index
            action = session.latest_action
            scene.draw(screen, session.history, frame_index, dt=frame_dt)
            hud.draw(
                screen,
                fonts,
                state=session.state,
                action=action,
                elapsed_time=session.time,
                status_text=self._status_text(),
                steps_label=f"{session.step_count}/{session.max_steps}",
                paused=self._paused,
                debug_forces=scene.show_force_vectors,
            )
            pygame.display.flip()

        pygame.quit()

    def _step_policy(self) -> None:
        """Advance one policy-controlled environment step."""

        if self._observation is None:
            raise RuntimeError("policy playback requires a valid initial observation")
        action = np.asarray(self._policy_fn(self._observation), dtype=np.float32)
        self._observation, _reward, terminated, truncated, _info = self._env.step(action)
        self._episode_finished = terminated or truncated

    def _reset_episode(self) -> None:
        """Reset the environment for a fresh playback episode."""

        self._observation, _info = self._env.reset()
        self._paused = False
        self._episode_finished = False

    def _handle_events(self) -> bool:
        """Process playback window events for the current frame."""

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
                self._reset_episode()
            elif (
                event.type == pygame.KEYDOWN
                and event.key == pygame.K_F3
                and self._scene is not None
            ):
                self._scene.toggle_force_vectors()
        return True

    def _handle_resize(self, width: int, height: int) -> None:
        """Resize the window and rebuild viewport-dependent renderer state."""

        self._viewport = self._viewport.resized(width, height)
        self._screen, self._viewport = create_display(self._viewport)
        self._rebuild_runtime()

    def _status_text(self) -> str:
        """Derive a short human-readable flight status for the HUD."""

        session = self._env.session
        if session.last_result is None:
            return "READY"
        if session.last_result.landed:
            return "LANDED"
        if session.last_result.crashed:
            return "CRASH"
        if self._episode_finished:
            return "DONE"
        if self._paused:
            return "PAUSED"
        return "POLICY"

    def _rebuild_runtime(self) -> None:
        """Recreate scene and HUD objects after initialization or resize."""

        if self._assets is None:
            raise RuntimeError("sprite assets must be loaded before building the runtime")
        camera = SceneCamera(self._env.params, self._viewport)
        self._scene = PygameSimulationScene(
            self._env.params,
            self._viewport,
            self._assets,
            camera,
            show_force_vectors=self._show_force_vectors,
            force_vector_scale_px_per_kn=self._force_vector_scale_px_per_kn,
        )
        self._hud = HeadsUpDisplay(self._env.params, self._viewport)
        self._fonts = HudFonts.create()

    def _require_runtime(
        self,
    ) -> tuple[pygame.Surface, PygameSimulationScene, HeadsUpDisplay, HudFonts]:
        """Return fully initialized pygame runtime objects."""

        if self._screen is None or self._scene is None or self._hud is None or self._fonts is None:
            raise RuntimeError("pygame runtime is not initialized")
        return self._screen, self._scene, self._hud, self._fonts
