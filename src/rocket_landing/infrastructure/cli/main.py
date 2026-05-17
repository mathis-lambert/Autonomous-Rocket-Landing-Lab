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
from rocket_landing.infrastructure.rendering.pygame.policy_app import PygamePolicySimulationApp
from rocket_landing.rl import (
    EnvConfig,
    RewardConfig,
    RocketLanderEnv,
    build_default_curriculum,
    find_stage_by_name,
    load_sac_model,
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
    parser.add_argument(
        "--controller",
        choices=("manual", "policy"),
        default="manual",
        help="Controller backend to drive the simulator",
    )
    parser.add_argument(
        "--policy-model",
        type=Path,
        help="Stable-Baselines3 policy checkpoint to replay in pygame",
    )
    parser.add_argument(
        "--policy-stage",
        help="Optional curriculum stage used to reset the RL environment for policy playback",
    )
    return parser


def run_live_session(args: argparse.Namespace) -> int:
    """Run the interactive pygame simulation."""

    if args.force_vector_scale <= 0.0:
        raise ValueError("force_vector_scale must be strictly positive")

    config = load_simulation_config(args.config)
    if args.controller == "manual":
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

    if args.policy_model is None:
        raise ValueError("--policy-model is required when --controller=policy")

    env = RocketLanderEnv(
        config.params,
        env_config=EnvConfig(dt=args.dt, max_episode_steps=args.steps),
        reward_config=RewardConfig(),
        default_initial_state=config.initial_state,
    )
    model = load_sac_model(args.policy_model)
    initial_state = config.initial_state
    stage_name_suffix = config.name
    if args.policy_stage is not None:
        stage = find_stage_by_name(build_default_curriculum(config.params), args.policy_stage)
        env.apply_curriculum_stage(stage)
        stage_name_suffix = stage.name
        initial_state = None
    app = PygamePolicySimulationApp(
        env,
        policy_fn=lambda obs: model.predict(obs, deterministic=True)[0],
        initial_state=initial_state,
        show_force_vectors=args.debug_forces,
        force_vector_scale_px_per_kn=args.force_vector_scale,
    )
    app.run(title=f"Rocket landing policy playback | {config.name} | {stage_name_suffix}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and launch the simulator."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return run_live_session(args)
