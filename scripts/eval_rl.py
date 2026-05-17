"""Evaluate a saved SAC policy with explicit stage alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rocket_landing.infrastructure.config import DEFAULT_CONFIG_PATH, load_simulation_config
from rocket_landing.rl import (
    build_default_curriculum,
    find_stage_by_name,
)
from rocket_landing.rl.campaign import benchmark_saved_model


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for policy evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a trained SAC rocket policy")
    parser.add_argument("model_path", type=Path, help="Path to a saved SAC .zip model")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Scenario YAML path",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=0, help="Evaluation seed")
    parser.add_argument(
        "--stage",
        type=str,
        default="full_envelope",
        help="Evaluation stage name",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device passed to SB3")
    return parser


def main() -> int:
    """Evaluate one saved model and print JSON report."""

    args = build_parser().parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    simulation_config = load_simulation_config(args.config)
    stages = build_default_curriculum(simulation_config.params)
    stage = find_stage_by_name(stages, args.stage)

    report = benchmark_saved_model(
        model_path=args.model_path,
        simulation_config=simulation_config,
        stage=stage,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "stage_name": stage.name,
                "report": report.as_dict(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
