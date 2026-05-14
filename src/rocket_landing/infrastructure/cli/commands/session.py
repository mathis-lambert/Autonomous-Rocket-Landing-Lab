"""Handlers for the live interactive session command."""

from __future__ import annotations

import argparse

from rocket_landing.infrastructure.config import load_simulation_config


def run_live_session(args: argparse.Namespace) -> int:
    """Run the interactive pygame application with one or more controllers.

    The live session owns the mutable simulation state while the controllers
    decide which actions to apply at each time step.
    """

    if args.force_vector_scale <= 0.0:
        raise ValueError("force_vector_scale must be strictly positive")

    from rocket_landing.application.control.baseline import BaselineLandingController
    from rocket_landing.application.use_cases.run_controlled_session import (
        ControlledSimulationSession,
    )
    from rocket_landing.infrastructure.rendering.pygame.live_app import (
        PygameLiveSimulationApp,
    )
    from rocket_landing.infrastructure.rendering.pygame.manual_controller import (
        PygameKeyboardManualController,
    )

    config = load_simulation_config(args.config)
    params = config.params

    session = ControlledSimulationSession(
        params=params,
        dt=args.dt,
        initial_state=config.initial_state,
        max_steps=args.steps,
    )
    controllers = [
        PygameKeyboardManualController(params),
        BaselineLandingController(params),
    ]
    app = PygameLiveSimulationApp(
        params,
        session,
        controllers=controllers,
        active_controller_name=args.controller,
        show_force_vectors=args.debug_forces,
        force_vector_scale_px_per_kn=args.force_vector_scale,
    )
    app.run(title="Rocket landing live session")
    return 0
