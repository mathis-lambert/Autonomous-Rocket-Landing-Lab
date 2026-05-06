from __future__ import annotations

import argparse
from collections.abc import Sequence

from rocket_landing.application.services.runtime import detect_acceleration_backend
from rocket_landing.application.use_cases.run_constant_action import RunConstantAction
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.infrastructure.rendering.matplotlib.trajectory_plotter import (
    MatplotlibTrajectoryPlotter,
)
from rocket_landing.infrastructure.rendering.pygame.app import PygameReplayApp


def _run_constant_action_demo(args: argparse.Namespace) -> int:
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

    if args.render_mode == "realtime":
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rocket landing simulation tools")
    parser.add_argument(
        "--throttle",
        type=float,
        default=0.85,
        help="Constant throttle command in [0, 1]",
    )
    parser.add_argument(
        "--gimbal",
        type=float,
        default=0.0,
        help="Constant gimbal command in radians",
    )
    parser.add_argument("--dt", type=float, default=0.02, help="Simulation time step in seconds")
    parser.add_argument(
        "--steps",
        type=int,
        default=3_000,
        help="Maximum number of simulation steps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for a saved figure instead of showing a window",
    )
    parser.add_argument(
        "--render-mode",
        choices=("realtime", "plot", "none"),
        default="realtime",
        help="Use the realtime renderer, a static plot, or disable rendering",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier used by the realtime renderer",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return _run_constant_action_demo(args)
