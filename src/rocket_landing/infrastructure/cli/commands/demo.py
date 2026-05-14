"""Handlers for the scripted constant-action demo command."""

from __future__ import annotations

import argparse

from rocket_landing.domain.models.action import Action
from rocket_landing.infrastructure.config import load_simulation_config


def run_constant_action_demo(args: argparse.Namespace) -> int:
    """Run a scripted constant-action scenario and optionally render it.

    This path is useful for deterministic debugging because the same action is
    applied at every simulation step.
    """

    if args.force_vector_scale <= 0.0:
        raise ValueError("force_vector_scale must be strictly positive")

    from rocket_landing.application.use_cases.run_constant_action import RunConstantAction
    from rocket_landing.infrastructure.rendering.matplotlib.trajectory_plotter import (
        MatplotlibTrajectoryPlotter,
    )
    from rocket_landing.infrastructure.rendering.pygame.app import PygameReplayApp

    config = load_simulation_config(args.config)
    params = config.params
    action = Action(
        throttle=args.throttle,
        engine_gimbal=args.gimbal,
        aero_steer=args.aero_steer,
    )
    run = RunConstantAction(
        params,
        args.dt,
        initial_state=config.initial_state,
    ).execute(action=action, max_steps=args.steps)
    final_state = run.final_result.state

    print(f"Scenario: {config.name}")
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
            show_force_vectors=args.debug_forces,
            force_vector_scale_px_per_kn=args.force_vector_scale,
        )
    elif args.render_mode == "plot" or args.output is not None:
        MatplotlibTrajectoryPlotter(params).render(
            run.history,
            title="Manual booster simulation",
            output_path=args.output,
        )
    return 0
