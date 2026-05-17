"""Train SAC policies with a clean, stage-aligned RL pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rocket_landing.infrastructure.config import DEFAULT_CONFIG_PATH, load_simulation_config
from rocket_landing.rl import (
    EvaluationConfig,
    TrainingConfig,
    build_default_curriculum,
    find_stage_by_name,
    train_sac,
)
from rocket_landing.rl.campaign import (
    apply_training_overrides,
    benchmark_saved_model,
    build_scheduler,
    build_training_env_factory,
    parse_seed_list,
    write_campaign_summary,
)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for training experiments."""

    parser = argparse.ArgumentParser(description="Train SAC rocket landing policies")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Scenario YAML path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/rl"),
        help="Base directory for artifacts",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Campaign name prefix",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from an existing SAC checkpoint",
    )
    parser.add_argument(
        "--resume-stage",
        type=str,
        default=None,
        help="Curriculum stage used when resuming with curriculum enabled",
    )
    parser.add_argument("--timesteps", type=int, default=250_000, help="Total training timesteps")
    parser.add_argument(
        "--segment-timesteps",
        type=int,
        default=10_000,
        help="Timesteps per train/eval segment",
    )
    parser.add_argument("--eval-episodes", type=int, default=50, help="Episodes per periodic eval")
    parser.add_argument("--n-envs", type=int, default=1, help="Parallel training environments")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override SAC learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override SAC batch size",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="Override SAC gradient steps",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="Override SAC learning_starts threshold",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=None,
        help="Override SAC replay buffer size",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0",
        help="Comma-separated integer seeds, e.g. '0,1,2'",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Train directly on fixed stage",
    )
    parser.add_argument(
        "--train-stage",
        type=str,
        default="full_envelope",
        help="Stage used when --no-curriculum is enabled",
    )
    parser.add_argument(
        "--benchmark-stage",
        type=str,
        default="full_envelope",
        help="Stage used for post-training benchmark",
    )
    parser.add_argument(
        "--benchmark-episodes",
        type=int,
        default=100,
        help="Episodes used for post-training benchmark",
    )
    parser.add_argument("--progress-bar", action="store_true", help="Enable SB3 progress bar")
    return parser


def main() -> int:
    """Run one training campaign over one or more seeds."""

    args = build_parser().parse_args()
    if args.timesteps <= 0 or args.segment_timesteps <= 0:
        raise ValueError("timesteps and segment-timesteps must be positive")
    if args.eval_episodes <= 0 or args.benchmark_episodes <= 0:
        raise ValueError("eval episodes must be positive")
    if args.n_envs <= 0:
        raise ValueError("n-envs must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0.0:
        raise ValueError("learning-rate must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.gradient_steps is not None and args.gradient_steps <= 0:
        raise ValueError("gradient-steps must be positive")
    if args.learning_starts is not None and args.learning_starts < 0:
        raise ValueError("learning-starts must be non-negative")
    if args.buffer_size is not None and args.buffer_size <= 0:
        raise ValueError("buffer-size must be positive")

    seeds = parse_seed_list(args.seeds)
    simulation_config = load_simulation_config(args.config)
    stages = build_default_curriculum(simulation_config.params)
    train_stage = find_stage_by_name(stages, args.train_stage)
    benchmark_stage = find_stage_by_name(stages, args.benchmark_stage)
    if args.resume_from is not None and not args.resume_from.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume_from}")
    if args.resume_from is not None and not args.no_curriculum and args.resume_stage is None:
        raise ValueError("resume-stage is required when resuming with curriculum enabled")

    campaign_name = args.run_name or f"{simulation_config.name}_sac_clean"
    campaign_dir = args.output_dir / campaign_name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    benchmark_success: list[float] = []
    benchmark_return: list[float] = []
    seed_runs: list[dict[str, object]] = []

    for seed in seeds:
        env_factory = build_training_env_factory(
            simulation_config,
            fixed_stage=train_stage if args.no_curriculum else None,
        )
        scheduler = build_scheduler(
            stages=stages,
            no_curriculum=args.no_curriculum,
            resume_from=args.resume_from,
            resume_stage=args.resume_stage,
        )

        run_name = f"{campaign_name}_seed{seed}"
        training_config = apply_training_overrides(
            TrainingConfig(
                total_timesteps=args.timesteps,
                segment_timesteps=args.segment_timesteps,
                n_envs=args.n_envs,
                seed=seed,
                progress_bar=args.progress_bar,
                evaluation=EvaluationConfig(
                    episodes=args.eval_episodes,
                    deterministic=True,
                    seed=seed,
                ),
            ),
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gradient_steps=args.gradient_steps,
            learning_starts=args.learning_starts,
            buffer_size=args.buffer_size,
        )
        summary = train_sac(
            env_factory=env_factory,
            output_dir=args.output_dir,
            run_name=run_name,
            config=training_config,
            curriculum=scheduler,
            initial_model_path=args.resume_from,
        )

        benchmark = benchmark_saved_model(
            model_path=summary.run_dir / "models" / "final_model.zip",
            simulation_config=simulation_config,
            stage=benchmark_stage,
            episodes=args.benchmark_episodes,
            seed=seed + 10_000,
        )

        benchmark_success.append(benchmark.metrics.success_rate)
        benchmark_return.append(benchmark.metrics.mean_return)
        seed_runs.append(
            {
                "seed": seed,
                "run_dir": summary.run_dir.as_posix(),
                "final_stage": summary.final_stage_name,
                "best_success_rate": summary.best_success_rate,
                "best_stage": summary.best_stage_name,
                "best_timesteps": summary.best_timesteps,
                "rollbacks": summary.rollbacks,
                "skipped_stages": list(summary.skipped_stages),
                "benchmark_stage": benchmark_stage.name,
                "benchmark_metrics": benchmark.metrics.as_dict(),
            }
        )

        print(f"seed={seed} run_dir={summary.run_dir}")
        print(f"seed={seed} final_stage={summary.final_stage_name}")
        print(
            f"seed={seed} best_stage={summary.best_stage_name} "
            f"at_timesteps={summary.best_timesteps}"
        )
        print(f"seed={seed} rollbacks={summary.rollbacks}")
        if summary.skipped_stages:
            print(f"seed={seed} skipped_stages={','.join(summary.skipped_stages)}")
        print(f"seed={seed} benchmark_success_rate={benchmark.metrics.success_rate:.3f}")

    summary_path = write_campaign_summary(
        campaign_dir=campaign_dir,
        campaign_name=campaign_name,
        scenario_name=simulation_config.name,
        seeds=seeds,
        benchmark_stage_name=benchmark_stage.name,
        benchmark_success=benchmark_success,
        benchmark_return=benchmark_return,
        seed_runs=seed_runs,
    )
    print(f"campaign_summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
