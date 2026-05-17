"""Backward-compatible manual pygame app wrapper."""

from __future__ import annotations

from rocket_landing.application.use_cases.run_controlled_session import ControlledSimulationSession
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.infrastructure.rendering.pygame.app import PygameSimulationApp
from rocket_landing.infrastructure.rendering.pygame.controllers.base import SimulationController
from rocket_landing.infrastructure.rendering.pygame.runtime import SessionRuntime
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


class PygameLiveSimulationApp(PygameSimulationApp):
    """Compatibility wrapper for manual sessions driven by the generic app."""

    def __init__(
        self,
        params: RocketParams,
        session: ControlledSimulationSession,
        *,
        controller: SimulationController,
        viewport: Viewport | None = None,
        show_force_vectors: bool = False,
        force_vector_scale_px_per_kn: float = 0.065,
    ) -> None:
        super().__init__(
            params,
            SessionRuntime(session),
            controller=controller,
            viewport=viewport,
            show_force_vectors=show_force_vectors,
            force_vector_scale_px_per_kn=force_vector_scale_px_per_kn,
        )
