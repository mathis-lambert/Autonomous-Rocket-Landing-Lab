"""Command-line entrypoints for demos and live simulation sessions.

The CLI is intentionally thin: it wires application-layer use cases and
infrastructure renderers together without embedding business logic.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rocket_landing.infrastructure.cli.commands import run_constant_action_demo, run_live_session
from rocket_landing.infrastructure.config import DEFAULT_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    """Build the root CLI parser and all subcommands.

    Returns:
        A fully configured argument parser exposing simulation workflows.
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
        help="Constant engine gimbal command in radians",
    )
    demo_parser.add_argument(
        "--aero-steer",
        type=float,
        default=0.0,
        help="Constant aerodynamic steering command in [-1, 1]",
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
    demo_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Scenario YAML to load",
    )
    demo_parser.add_argument(
        "--debug-forces",
        action="store_true",
        help="Start the replay with force-vector debug overlay enabled",
    )
    demo_parser.add_argument(
        "--force-vector-scale",
        type=float,
        default=0.065,
        help="Debug overlay scale in pixels per kilonewton",
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
        default=50_000,
        help="Maximum number of simulation steps before auto-stop",
    )
    session_parser.add_argument(
        "--controller",
        choices=("manual", "baseline"),
        default="manual",
        help="Initial controller used by the live session",
    )
    session_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Scenario YAML to load",
    )
    session_parser.add_argument(
        "--debug-forces",
        action="store_true",
        help="Start the live session with force-vector debug overlay enabled",
    )
    session_parser.add_argument(
        "--force-vector-scale",
        type=float,
        default=0.065,
        help="Debug overlay scale in pixels per kilonewton",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected command."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "session":
        return run_live_session(args)
    if args.command == "demo":
        return run_constant_action_demo(args)
    parser.error("a subcommand is required")
    return 2
