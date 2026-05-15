"""Train a SAC agent on the rocket landing environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from rocket_landing.application.rl.config import EvaluationConfig, RewardConfig, RLEnvConfig
from rocket_landing.application.rl.curriculum.scheduler import CurriculumScheduler
from rocket_landing.application.rl.curriculum.stages import build_default_curriculum
from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.infrastructure.config import DEFAULT_CONFIG_PATH, load_simulation_config
from rocket_landing.infrastructure.rl.sb3 import SB3Trainer, SB3TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for SAC training."""

    parser = argparse.ArgumentParser(description="Train a SAC rocket landing policy")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Scenario YAML to load",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl"),
        help="Directory where training artifacts will be stored",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional explicit training run name",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=250_000,
        help="Total SAC training timesteps",
    )
    parser.add_argument(
        "--segment-timesteps",
        type=int,
        default=10_000,
        help="Timesteps per train/eval segment",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Episodes per evaluation pass",
    )
    parser.add_argument("--n-envs", type=int, default=1, help="Parallel training environments")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed")
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable staged reset distributions and train on the full envelope immediately",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show the SB3 progress bar during learning",
    )
    return parser


def main() -> int:
    """Parse arguments and launch SAC training."""

    args = build_parser().parse_args()
    if args.timesteps <= 0 or args.segment_timesteps <= 0 or args.eval_episodes <= 0:
        raise ValueError("timesteps, segment-timesteps, and eval-episodes must be positive")
    if args.n_envs <= 0:
        raise ValueError("n-envs must be strictly positive")

    simulation_config = load_simulation_config(args.config)
    env_config = RLEnvConfig()
    reward_config = RewardConfig()

    def env_factory() -> RocketLanderEnv:
        return RocketLanderEnv(
            simulation_config.params,
            env_config=env_config,
            reward_config=reward_config,
            default_initial_state=simulation_config.initial_state,
        )

    curriculum = None
    if not args.no_curriculum:
        curriculum = CurriculumScheduler(build_default_curriculum(simulation_config.params))

    trainer = SB3Trainer(
        SB3TrainerConfig(
            total_timesteps=args.timesteps,
            train_segment_timesteps=args.segment_timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            progress_bar=args.progress_bar,
            evaluation=EvaluationConfig(
                episodes=args.eval_episodes,
                deterministic=True,
                seed=args.seed,
            ),
        )
    )
    summary = trainer.train(
        env_factory=env_factory,
        output_dir=args.output_dir,
        run_name=args.run_name or f"{simulation_config.name}_sac",
        curriculum=curriculum,
    )
    print(f"run_dir={summary.run_dir}")
    print(f"total_timesteps={summary.total_timesteps}")
    print(f"best_success_rate={summary.best_success_rate:.3f}")
    print(f"final_stage={summary.final_stage_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
