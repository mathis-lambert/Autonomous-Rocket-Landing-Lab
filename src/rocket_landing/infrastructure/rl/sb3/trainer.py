"""SB3 trainer orchestrating learning, evaluation and curriculum updates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rocket_landing.application.rl.curriculum.scheduler import CurriculumScheduler
from rocket_landing.application.rl.env import RocketLanderEnv
from rocket_landing.application.rl.evaluation.evaluator import EvaluationReport, evaluate_policy
from rocket_landing.infrastructure.rl.sb3.agent_factory import build_sac_model
from rocket_landing.infrastructure.rl.sb3.callbacks import build_training_callbacks
from rocket_landing.infrastructure.rl.sb3.checkpointing import (
    prepare_training_paths,
    save_best_model,
    save_checkpoint,
    save_evaluation_report,
    save_final_model,
    save_latest_model,
    save_run_manifest,
)
from rocket_landing.infrastructure.rl.sb3.train_config import SB3TrainerConfig

EnvFactory = Callable[[], RocketLanderEnv]


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Compact summary returned when training finishes."""

    total_timesteps: int
    best_success_rate: float
    final_stage_name: str
    evaluations_run: int
    run_dir: Path


class SB3Trainer:
    """Train a SAC agent while periodically evaluating curriculum progress."""

    def __init__(self, config: SB3TrainerConfig | None = None) -> None:
        self._config = config or SB3TrainerConfig()

    @property
    def config(self) -> SB3TrainerConfig:
        """Expose the immutable training configuration."""

        return self._config

    def train(
        self,
        *,
        env_factory: EnvFactory,
        output_dir: Path,
        run_name: str,
        curriculum: CurriculumScheduler | None = None,
    ) -> TrainingSummary:
        """Train a SAC model, evaluate it, and persist useful artifacts."""

        training_paths = prepare_training_paths(output_dir, run_name)
        train_env = self._build_train_env(env_factory)
        eval_env = env_factory()

        if curriculum is not None:
            self._apply_stage(train_env, eval_env, curriculum.current_stage)

        save_run_manifest(
            training_paths,
            {
                "trainer_config": asdict(self._config),
                "curriculum_stages": []
                if curriculum is None
                else [stage.name for stage in curriculum.stages],
            },
        )

        model = build_sac_model(
            train_env,
            config=self._config,
            tensorboard_log_dir=training_paths.tensorboard_dir,
        )
        callbacks = build_training_callbacks(
            stage_name_provider=lambda: eval_env.current_stage_name,
        )

        total_trained = 0
        next_checkpoint = self._config.checkpoints.save_freq_timesteps
        evaluations_run = 0
        best_report_score = (float("-inf"), float("-inf"), float("-inf"))
        best_success_rate = float("-inf")

        while total_trained < self._config.total_timesteps:
            segment = min(
                self._config.train_segment_timesteps,
                self._config.total_timesteps - total_trained,
            )
            model.learn(
                total_timesteps=segment,
                reset_num_timesteps=False,
                progress_bar=self._config.progress_bar,
                callback=callbacks,
            )
            total_trained += segment

            report = evaluate_policy(
                eval_env,
                lambda observation: model.predict(
                    observation,
                    deterministic=self._config.evaluation.deterministic,
                )[0],
                config=self._config.evaluation,
            )
            evaluations_run += 1
            save_evaluation_report(
                training_paths,
                timesteps=total_trained,
                stage_name=eval_env.current_stage_name,
                report=report.as_dict(),
            )

            if self._config.checkpoints.save_latest_model:
                save_latest_model(model, training_paths)
            while next_checkpoint <= total_trained:
                save_checkpoint(model, training_paths, next_checkpoint)
                next_checkpoint += self._config.checkpoints.save_freq_timesteps

            report_score = self._report_score(report)
            if self._config.checkpoints.save_best_model and report_score > best_report_score:
                best_report_score = report_score
                best_success_rate = report.metrics.success_rate
                save_best_model(model, training_paths)
            elif best_success_rate == float("-inf"):
                best_success_rate = report.metrics.success_rate

            if curriculum is not None and curriculum.maybe_advance(report.metrics):
                self._apply_stage(train_env, eval_env, curriculum.current_stage)

        save_final_model(model, training_paths)
        train_env.close()
        eval_env.close()
        return TrainingSummary(
            total_timesteps=total_trained,
            best_success_rate=best_success_rate,
            final_stage_name=eval_env.current_stage_name,
            evaluations_run=evaluations_run,
            run_dir=training_paths.run_dir,
        )

    def _build_train_env(self, env_factory: EnvFactory) -> Any:
        """Create the vectorized training environment."""

        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

        env_fns = [env_factory for _ in range(self._config.n_envs)]
        return VecMonitor(DummyVecEnv(env_fns))

    @staticmethod
    def _apply_stage(train_env: Any, eval_env: RocketLanderEnv, stage: Any) -> None:
        """Apply one curriculum stage to both training and evaluation envs."""

        train_env.env_method("apply_curriculum_stage", stage)
        eval_env.apply_curriculum_stage(stage)

    @staticmethod
    def _report_score(report: EvaluationReport) -> tuple[float, float, float]:
        """Rank evaluation reports for best-model checkpointing."""

        return (
            report.metrics.success_rate,
            report.metrics.mean_return,
            -report.metrics.mean_abs_final_dx,
        )
