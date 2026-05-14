"""Command-line entrypoint for the live booster simulation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rocket_landing.application.use_cases.run_controlled_session import ControlledSimulationSession
from rocket_landing.infrastructure.config import DEFAULT_CONFIG_PATH, load_simulation_config
from rocket_landing.infrastructure.rendering.pygame.live_app import PygameLiveSimulationApp
from rocket_landing.infrastructure.rendering.pygame.manual_controller import (
    PygameKeyboardManualController,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the live simulator."""

    parser = argparse.ArgumentParser(description="Rocket landing live simulator")
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Simulation time step in seconds",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000,
        help="Maximum number of simulation steps before auto-stop",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Scenario YAML to load",
    )
    parser.add_argument(
        "--debug-forces",
        action="store_true",
        help="Start the live session with force-vector debug overlay enabled",
    )
    parser.add_argument(
        "--force-vector-scale",
        type=float,
        default=0.065,
        help="Debug overlay scale in pixels per kilonewton",
    )
    return parser


def run_live_session(args: argparse.Namespace) -> int:
    """Run the interactive pygame simulation."""

    if args.force_vector_scale <= 0.0:
        raise ValueError("force_vector_scale must be strictly positive")

    config = load_simulation_config(args.config)
    session = ControlledSimulationSession(
        params=config.params,
        dt=args.dt,
        initial_state=config.initial_state,
        max_steps=args.steps,
    )
    controller = PygameKeyboardManualController(config.params, config.controls)
    app = PygameLiveSimulationApp(
        config.params,
        session,
        controller=controller,
        show_force_vectors=args.debug_forces,
        force_vector_scale_px_per_kn=args.force_vector_scale,
    )
    app.run(title=f"Rocket landing live session | {config.name}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and launch the simulator."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return run_live_session(args)
