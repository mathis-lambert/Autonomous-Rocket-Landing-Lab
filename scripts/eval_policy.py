"""Evaluate a saved SAC policy on the rocket landing environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocket_landing.application.rl.config import EvaluationConfig, RewardConfig, RLEnvConfig
from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.application.rl.evaluation.evaluator import evaluate_policy
from rocket_landing.infrastructure.config import DEFAULT_CONFIG_PATH, load_simulation_config
from rocket_landing.infrastructure.rl.sb3 import load_sac_model


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for policy evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a trained SAC rocket policy")
    parser.add_argument("model_path", type=Path, help="Path to a saved `.zip` SAC model")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Scenario YAML to load",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=None, help="Optional evaluation seed")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions instead of deterministic evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device passed to SB3 load",
    )
    return parser


def main() -> int:
    """Parse arguments and evaluate the requested policy."""

    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be strictly positive")

    simulation_config = load_simulation_config(args.config)
    env = RocketLanderEnv(
        simulation_config.params,
        env_config=RLEnvConfig(),
        reward_config=RewardConfig(),
        default_initial_state=simulation_config.initial_state,
    )
    model = load_sac_model(args.model_path, device=args.device)
    report = evaluate_policy(
        env,
        lambda observation: model.predict(observation, deterministic=not args.stochastic)[0],
        config=EvaluationConfig(
            episodes=args.episodes,
            deterministic=not args.stochastic,
            seed=args.seed,
        ),
    )
    print(json.dumps(report.as_dict(), indent=2))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
