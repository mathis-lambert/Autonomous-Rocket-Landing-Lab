"""Backward-compatible policy pygame app wrapper."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from rocket_landing.infrastructure.rendering.pygame.app import PygameSimulationApp
from rocket_landing.infrastructure.rendering.pygame.controllers.policy import PolicyController
from rocket_landing.infrastructure.rendering.pygame.runtime import EnvironmentRuntime
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport
from rocket_landing.rl.env import RocketLanderEnv, build_observation

PolicyFn = Callable[[np.ndarray], np.ndarray]


class PygamePolicySimulationApp(PygameSimulationApp):
    """Drive the simulator with a learned policy while reusing the generic app."""

    def __init__(
        self,
        env: RocketLanderEnv,
        *,
        policy_fn: PolicyFn,
        viewport: Viewport | None = None,
        show_force_vectors: bool = False,
        force_vector_scale_px_per_kn: float = 0.065,
    ) -> None:
        controller = PolicyController(
            env.params,
            policy_fn=policy_fn,
            observation_fn=lambda state, params=env.params: build_observation(state, params),
        )
        super().__init__(
            env.params,
            EnvironmentRuntime(env),
            controller=controller,
            viewport=viewport,
            show_force_vectors=show_force_vectors,
            force_vector_scale_px_per_kn=force_vector_scale_px_per_kn,
        )
