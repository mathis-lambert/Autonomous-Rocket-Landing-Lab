"""Play a trained SAC policy in headless mode or with the pygame renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocket_landing.application.rl.config import EvaluationConfig, RewardConfig, RLEnvConfig
from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.application.rl.evaluation.evaluator import evaluate_policy
from rocket_landing.infrastructure.config import DEFAULT_CONFIG_PATH, load_simulation_config
from rocket_landing.infrastructure.rendering.pygame.policy_app import PygamePolicySimulationApp
from rocket_landing.infrastructure.rl.sb3 import load_sac_model


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for policy playback."""

    parser = argparse.ArgumentParser(description="Play a trained SAC rocket policy")
    parser.add_argument("model_path", type=Path, help="Path to a saved `.zip` SAC model")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Scenario YAML to load",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional reset seed")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic playback",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run one episode without opening pygame and print the summary as JSON",
    )
    parser.add_argument(
        "--debug-forces",
        action="store_true",
        help="Start the pygame playback with force-vector debug overlay enabled",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device passed to SB3 load",
    )
    return parser


def main() -> int:
    """Parse arguments and launch policy playback."""

    args = build_parser().parse_args()
    simulation_config = load_simulation_config(args.config)
    env = RocketLanderEnv(
        simulation_config.params,
        env_config=RLEnvConfig(),
        reward_config=RewardConfig(),
        default_initial_state=simulation_config.initial_state,
    )
    model = load_sac_model(args.model_path, device=args.device)

    def policy_fn(observation: object) -> object:
        return model.predict(observation, deterministic=not args.stochastic)[0]

    if args.headless:
        report = evaluate_policy(
            env,
            policy_fn,
            config=EvaluationConfig(
                episodes=1,
                deterministic=not args.stochastic,
                seed=args.seed,
            ),
        )
        print(json.dumps(report.as_dict(), indent=2))
        env.close()
        return 0

    app = PygamePolicySimulationApp(
        env,
        policy_fn=policy_fn,
        show_force_vectors=args.debug_forces,
    )
    app.run(title=f"Rocket landing policy playback | {simulation_config.name}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
