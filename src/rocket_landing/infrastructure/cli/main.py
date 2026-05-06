"""Command-line entrypoints for demos and live simulation sessions.

The CLI is intentionally thin: it wires application-layer use cases and
infrastructure renderers together without embedding business logic.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from rocket_landing.application.control.baseline import BaselineLandingController
from rocket_landing.application.services.runtime import detect_acceleration_backend
from rocket_landing.application.use_cases.run_constant_action import RunConstantAction
from rocket_landing.application.use_cases.run_controlled_session import ControlledSimulationSession
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.infrastructure.rendering.matplotlib.trajectory_plotter import (
    MatplotlibTrajectoryPlotter,
)
from rocket_landing.infrastructure.rendering.pygame.app import PygameReplayApp
from rocket_landing.infrastructure.rendering.pygame.live_app import PygameLiveSimulationApp
from rocket_landing.infrastructure.rendering.pygame.manual_controller import (
    PygameKeyboardManualController,
)


def _run_constant_action_demo(args: argparse.Namespace) -> int:
    """Run a scripted constant-action scenario and optionally render it.

    This path is useful for deterministic debugging because the same action is
    applied at every simulation step.
    """

    params = RocketParams()
    action = Action(throttle=args.throttle, gimbal=args.gimbal)
    run = RunConstantAction(params, args.dt).execute(action=action, max_steps=args.steps)
    final_state = run.final_result.state

    print(f"Acceleration backend: {detect_acceleration_backend()}")
    print(f"Simulation time: {run.final_time:.2f} s")
    print(f"Final position: x={final_state.x:.2f} m, z={final_state.z:.2f} m")
    print(f"Final velocity: vx={final_state.vx:.2f} m/s, vz={final_state.vz:.2f} m/s")
    print(f"Final attitude: theta={final_state.theta:.3f} rad, omega={final_state.omega:.3f} rad/s")
    print(f"Remaining fuel: {final_state.fuel:.2f} kg")
    print(f"Landed: {run.final_result.landed} | Crashed: {run.final_result.crashed}")

    if args.render_mode == "replay":
        PygameReplayApp(params).run(
            run.history,
            title="Manual booster simulation",
            playback_speed=args.playback_speed,
        )
    elif args.render_mode == "plot" or args.output is not None:
        MatplotlibTrajectoryPlotter(params).render(
            run.history,
            title="Manual booster simulation",
            output_path=args.output,
        )
    return 0


def _run_live_session(args: argparse.Namespace) -> int:
    """Run the interactive pygame application with one or more controllers.

    The live session owns the mutable simulation state while the controllers
    decide which actions to apply at each time step.
    """

    params = RocketParams()
    session = ControlledSimulationSession(params=params, dt=args.dt, max_steps=args.steps)
    controllers = [
        PygameKeyboardManualController(params),
        BaselineLandingController(params),
    ]
    app = PygameLiveSimulationApp(
        params,
        session,
        controllers=controllers,
        active_controller_name=args.controller,
    )
    app.run(title="Rocket landing live session")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser and all subcommands.

    Returns:
        A fully configured argument parser exposing the ``demo`` and
        ``session`` workflows.
    """

    parser = argparse.ArgumentParser(description="Rocket landing simulation tools")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="Replay a constant-action simulation")
    demo_parser.add_argument(
        "--throttle",
        type=float,
        default=0.85,
        help="Constant throttle command in [0, 1]",
    )
    demo_parser.add_argument(
        "--gimbal",
        type=float,
        default=0.0,
        help="Constant gimbal command in radians",
    )
    demo_parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Simulation time step in seconds",
    )
    demo_parser.add_argument(
        "--steps",
        type=int,
        default=3_000,
        help="Maximum number of simulation steps",
    )
    demo_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for a saved figure instead of showing a window",
    )
    demo_parser.add_argument(
        "--render-mode",
        choices=("replay", "plot", "none"),
        default="replay",
        help="Use the replay renderer, a static plot, or disable rendering",
    )
    demo_parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier used by the replay renderer",
    )

    session_parser = subparsers.add_parser("session", help="Run a live interactive simulation")
    session_parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Simulation time step in seconds",
    )
    session_parser.add_argument(
        "--steps",
        type=int,
        default=10_000,
        help="Maximum number of simulation steps before auto-stop",
    )
    session_parser.add_argument(
        "--controller",
        choices=("manual", "baseline"),
        default="manual",
        help="Initial controller used by the live session",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, normalize shortcuts and dispatch to the selected command.

    A few shortcuts are supported for convenience:
    - no arguments starts an interactive ``session``
    - leading options without a subcommand are interpreted as ``demo`` options
    """

    effective_argv = None if argv is None else list(argv)
    if effective_argv == []:
        effective_argv = ["session"]
    elif effective_argv is not None and effective_argv[0].startswith("-"):
        effective_argv = ["demo", *effective_argv]

    parser = build_parser()
    args = parser.parse_args(effective_argv)
    if args.command == "session":
        return _run_live_session(args)
    if args.command == "demo":
        return _run_constant_action_demo(args)
    raise ValueError(f"unsupported command: {args.command}")
