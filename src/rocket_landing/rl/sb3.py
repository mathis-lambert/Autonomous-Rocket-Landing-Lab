"""SB3 SAC training and persistence helpers."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rocket_landing.rl.curriculum import CurriculumScheduler, meets_promotion_criteria
from rocket_landing.rl.env import RocketLanderEnv
from rocket_landing.rl.evaluation import EvaluationReport, evaluate_policy
from rocket_landing.rl.types import CurriculumStage, TrainingConfig

EnvFactory = Callable[[], RocketLanderEnv]


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    """Filesystem paths for one training run."""

    run_dir: Path
    checkpoints_dir: Path
    evaluations_dir: Path
    tensorboard_dir: Path
    models_dir: Path


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Compact training outcome summary."""

    total_timesteps: int
    best_success_rate: float
    best_stage_name: str
    best_timesteps: int
    final_stage_name: str
    evaluations_run: int
    rollbacks: int
    skipped_stages: tuple[str, ...]
    run_dir: Path


@dataclass(frozen=True, slots=True)
class StageBest:
    """Best checkpoint observed for one curriculum stage."""

    score: tuple[float, float, float]
    success_rate: float
    timesteps: int
    path: Path


def prepare_training_paths(output_dir: Path, run_name: str) -> TrainingPaths:
    """Create directory structure for one run."""

    run_dir = output_dir / run_name
    checkpoints_dir = run_dir / "checkpoints"
    evaluations_dir = run_dir / "evaluations"
    tensorboard_dir = run_dir / "tensorboard"
    models_dir = run_dir / "models"
    for path in (run_dir, checkpoints_dir, evaluations_dir, tensorboard_dir, models_dir):
        path.mkdir(parents=True, exist_ok=True)
    return TrainingPaths(
        run_dir=run_dir,
        checkpoints_dir=checkpoints_dir,
        evaluations_dir=evaluations_dir,
        tensorboard_dir=tensorboard_dir,
        models_dir=models_dir,
    )


def load_sac_model(model_path: Path, *, env: Any | None = None, device: str = "auto") -> Any:
    """Load SAC model from disk."""

    from stable_baselines3 import SAC

    return SAC.load(model_path.as_posix(), env=env, device=device)


def _build_sac_model(env: Any, *, config: TrainingConfig, tensorboard_log_dir: Path | None) -> Any:
    """Construct one SAC model from trainer config."""

    from stable_baselines3 import SAC

    tensorboard_log = None
    if tensorboard_log_dir is not None and importlib.util.find_spec("tensorboard") is not None:
        tensorboard_log = tensorboard_log_dir.as_posix()

    return SAC(
        policy=config.sac.policy,
        env=env,
        learning_rate=config.sac.learning_rate,
        buffer_size=config.sac.buffer_size,
        learning_starts=config.sac.learning_starts,
        batch_size=config.sac.batch_size,
        tau=config.sac.tau,
        gamma=config.sac.gamma,
        train_freq=config.sac.train_freq,
        gradient_steps=config.sac.gradient_steps,
        ent_coef=config.sac.ent_coef,
        verbose=config.verbose,
        device=config.sac.device,
        seed=config.seed,
        tensorboard_log=tensorboard_log,
    )


def _apply_sac_runtime_overrides(model: Any, *, config: TrainingConfig) -> None:
    """Apply safe runtime overrides to a loaded SAC model before resuming training."""

    optimizers: list[Any] = []
    model.learning_rate = config.sac.learning_rate
    model._setup_lr_schedule()

    model.buffer_size = config.sac.buffer_size
    model.learning_starts = config.sac.learning_starts
    model.batch_size = config.sac.batch_size
    model.tau = config.sac.tau
    model.gamma = config.sac.gamma
    model.train_freq = config.sac.train_freq
    if hasattr(model, "_convert_train_freq"):
        model._convert_train_freq()
    model.gradient_steps = config.sac.gradient_steps

    if hasattr(model, "actor") and hasattr(model.actor, "optimizer"):
        optimizers.append(model.actor.optimizer)
    if hasattr(model, "critic") and hasattr(model.critic, "optimizer"):
        optimizers.append(model.critic.optimizer)
    if hasattr(model, "ent_coef_optimizer") and model.ent_coef_optimizer is not None:
        optimizers.append(model.ent_coef_optimizer)
    if optimizers:
        learning_rate = float(model.lr_schedule(1.0))
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate


def _save_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _save_evaluation_report(
    paths: TrainingPaths,
    *,
    timesteps: int,
    stage_name: str,
    report: EvaluationReport,
) -> None:
    _save_json(
        paths.evaluations_dir / f"eval_step_{timesteps:09d}.json",
        {
            "timesteps": timesteps,
            "stage_name": stage_name,
            "report": report.as_dict(),
        },
    )


def _save_probe_report(
    paths: TrainingPaths,
    *,
    timesteps: int,
    from_stage_name: str,
    stage_name: str,
    report: EvaluationReport,
) -> None:
    _save_json(
        paths.evaluations_dir / f"probe_step_{timesteps:09d}_{stage_name}.json",
        {
            "timesteps": timesteps,
            "from_stage_name": from_stage_name,
            "stage_name": stage_name,
            "report": report.as_dict(),
        },
    )


def _report_score(report: EvaluationReport) -> tuple[float, float, float]:
    return (
        report.metrics.success_rate,
        report.metrics.mean_return,
        -report.metrics.mean_abs_final_dx,
    )


def _stage_model_path(paths: TrainingPaths, stage_name: str) -> Path:
    return paths.models_dir / f"best_{stage_name}.zip"


def _record_stage_best(
    *,
    model: Any,
    paths: TrainingPaths,
    stage: CurriculumStage,
    report: EvaluationReport,
    timesteps: int,
    stage_bests: dict[str, StageBest],
) -> None:
    score = _report_score(report)
    existing = stage_bests.get(stage.name)
    if existing is not None and score <= existing.score:
        return
    path = _stage_model_path(paths, stage.name)
    model.save(path.as_posix())
    stage_bests[stage.name] = StageBest(
        score=score,
        success_rate=report.metrics.success_rate,
        timesteps=timesteps,
        path=path,
    )


def _should_rollback(
    *,
    stage_name: str,
    report: EvaluationReport,
    config: TrainingConfig,
    stage_bests: dict[str, StageBest],
    rollback_counts: dict[str, int],
) -> bool:
    best = stage_bests.get(stage_name)
    if best is None:
        return False
    if best.success_rate < config.curriculum_guard.rollback_min_best_success_rate:
        return False
    if rollback_counts.get(stage_name, 0) >= config.curriculum_guard.max_rollbacks_per_stage:
        return False
    success_drop = best.success_rate - report.metrics.success_rate
    return success_drop >= config.curriculum_guard.rollback_success_drop


def _make_vec_env(env_factory: EnvFactory, n_envs: int) -> Any:
    """Build a vectorized environment suited to the requested parallelism."""

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    env_fns = [env_factory for _ in range(n_envs)]
    if n_envs == 1:
        return VecMonitor(DummyVecEnv(env_fns))
    start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
    return VecMonitor(SubprocVecEnv(env_fns, start_method=start_method))


def train_sac(
    *,
    env_factory: EnvFactory,
    output_dir: Path,
    run_name: str,
    config: TrainingConfig,
    curriculum: CurriculumScheduler | None,
    initial_model_path: Path | None = None,
) -> TrainingSummary:
    """Train SAC and persist checkpoints and evaluation reports."""

    if config.total_timesteps <= 0 or config.segment_timesteps <= 0:
        raise ValueError("timesteps and segment timesteps must be positive")
    if config.n_envs <= 0:
        raise ValueError("n_envs must be positive")

    paths = prepare_training_paths(output_dir, run_name)
    _save_json(
        paths.run_dir / "manifest.json",
        {
            "trainer_config": asdict(config),
            "initial_model_path": (
                None if initial_model_path is None else initial_model_path.as_posix()
            ),
            "curriculum_stages": []
            if curriculum is None
            else [stage.name for stage in curriculum.stages],
        },
    )

    train_env = _make_vec_env(env_factory, config.n_envs)
    eval_env = env_factory()

    if curriculum is not None:
        train_env.env_method("apply_curriculum_stage", curriculum.current_stage)
        eval_env.apply_curriculum_stage(curriculum.current_stage)

    if initial_model_path is None:
        model = _build_sac_model(
            train_env,
            config=config,
            tensorboard_log_dir=paths.tensorboard_dir,
        )
    else:
        model = load_sac_model(initial_model_path, env=train_env, device=config.sac.device)
        _apply_sac_runtime_overrides(model, config=config)

    total_trained = 0
    next_checkpoint = config.checkpoints.save_freq_timesteps
    evaluations_run = 0
    best_success_rate = float("-inf")
    best_stage_name = eval_env.current_stage_name
    best_timesteps = 0
    best_report_score = (float("-inf"), float("-inf"), float("-inf"))
    stage_bests: dict[str, StageBest] = {}
    rollback_counts: dict[str, int] = {}
    rollback_events: list[dict[str, object]] = []
    skipped_stages: list[str] = []

    while total_trained < config.total_timesteps:
        segment = min(config.segment_timesteps, config.total_timesteps - total_trained)
        model.learn(
            total_timesteps=segment,
            reset_num_timesteps=False,
            progress_bar=config.progress_bar,
        )
        total_trained += segment
        report = evaluate_policy(
            eval_env,
            lambda obs, local_model=model: local_model.predict(
                obs,
                deterministic=config.evaluation.deterministic,
            )[0],
            config=config.evaluation,
        )
        evaluations_run += 1
        _save_evaluation_report(
            paths,
            timesteps=total_trained,
            stage_name=eval_env.current_stage_name,
            report=report,
        )
        current_stage = curriculum.current_stage if curriculum is not None else None
        if current_stage is not None:
            _record_stage_best(
                model=model,
                paths=paths,
                stage=current_stage,
                report=report,
                timesteps=total_trained,
                stage_bests=stage_bests,
            )

        if config.checkpoints.save_latest_model:
            model.save((paths.models_dir / "latest_model.zip").as_posix())
        while next_checkpoint <= total_trained:
            model.save((paths.checkpoints_dir / f"sac_step_{next_checkpoint:09d}.zip").as_posix())
            next_checkpoint += config.checkpoints.save_freq_timesteps

        score = _report_score(report)
        if config.checkpoints.save_best_model and score > best_report_score:
            best_report_score = score
            best_success_rate = report.metrics.success_rate
            best_stage_name = eval_env.current_stage_name
            best_timesteps = total_trained
            model.save((paths.models_dir / "best_model.zip").as_posix())
        elif best_success_rate == float("-inf"):
            best_success_rate = report.metrics.success_rate
            best_stage_name = eval_env.current_stage_name
            best_timesteps = total_trained

        if current_stage is not None and _should_rollback(
            stage_name=current_stage.name,
            report=report,
            config=config,
            stage_bests=stage_bests,
            rollback_counts=rollback_counts,
        ):
            best = stage_bests[current_stage.name]
            rollback_counts[current_stage.name] = rollback_counts.get(current_stage.name, 0) + 1
            rollback_events.append(
                {
                    "timesteps": total_trained,
                    "stage_name": current_stage.name,
                    "success_rate": report.metrics.success_rate,
                    "restored_timesteps": best.timesteps,
                    "restored_success_rate": best.success_rate,
                }
            )
            shutil.copy2(best.path, paths.models_dir / "latest_model.zip")
            model = load_sac_model(best.path, env=train_env, device=config.sac.device)
            train_env.env_method("apply_curriculum_stage", current_stage)
            eval_env.apply_curriculum_stage(current_stage)
            _save_json(paths.run_dir / "rollback_events.json", {"events": rollback_events})
            continue

        if current_stage is not None and meets_promotion_criteria(current_stage, report.metrics):
            while curriculum is not None and curriculum.next_stage is not None:
                from_stage_name = curriculum.current_stage.name
                next_stage = curriculum.next_stage
                probe_report = None
                if config.curriculum_guard.enable_next_stage_probe:
                    eval_env.apply_curriculum_stage(next_stage)
                    probe_report = evaluate_policy(
                        eval_env,
                        lambda obs, local_model=model: local_model.predict(
                            obs,
                            deterministic=config.evaluation.deterministic,
                        )[0],
                        config=config.evaluation,
                    )
                    evaluations_run += 1
                    _save_probe_report(
                        paths,
                        timesteps=total_trained,
                        from_stage_name=from_stage_name,
                        stage_name=next_stage.name,
                        report=probe_report,
                    )
                    _record_stage_best(
                        model=model,
                        paths=paths,
                        stage=next_stage,
                        report=probe_report,
                        timesteps=total_trained,
                        stage_bests=stage_bests,
                    )

                curriculum.advance()
                if probe_report is None:
                    break
                if not meets_promotion_criteria(next_stage, probe_report.metrics):
                    break
                skipped_stages.append(next_stage.name)

            if curriculum is not None:
                train_env.env_method("apply_curriculum_stage", curriculum.current_stage)
                eval_env.apply_curriculum_stage(curriculum.current_stage)

    model.save((paths.models_dir / "final_model.zip").as_posix())
    train_env.close()
    eval_env.close()
    if rollback_events:
        _save_json(paths.run_dir / "rollback_events.json", {"events": rollback_events})
    return TrainingSummary(
        total_timesteps=total_trained,
        best_success_rate=best_success_rate,
        best_stage_name=best_stage_name,
        best_timesteps=best_timesteps,
        final_stage_name=eval_env.current_stage_name,
        evaluations_run=evaluations_run,
        rollbacks=sum(rollback_counts.values()),
        skipped_stages=tuple(skipped_stages),
        run_dir=paths.run_dir,
    )
