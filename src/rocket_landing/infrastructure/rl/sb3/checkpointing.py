"""Model persistence and training artifact helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    """Resolved filesystem locations for one training run."""

    run_dir: Path
    checkpoints_dir: Path
    evaluations_dir: Path
    tensorboard_dir: Path
    models_dir: Path


def prepare_training_paths(output_dir: Path, run_name: str) -> TrainingPaths:
    """Create and return the directory layout for one training run."""

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


def save_checkpoint(model: Any, paths: TrainingPaths, timesteps: int) -> Path:
    """Save a numbered intermediate checkpoint."""

    path = paths.checkpoints_dir / f"sac_step_{timesteps:09d}.zip"
    model.save(path.as_posix())
    return path


def save_latest_model(model: Any, paths: TrainingPaths) -> Path:
    """Save the latest model snapshot."""

    path = paths.models_dir / "latest_model.zip"
    model.save(path.as_posix())
    return path


def save_best_model(model: Any, paths: TrainingPaths) -> Path:
    """Save the best-known model snapshot."""

    path = paths.models_dir / "best_model.zip"
    model.save(path.as_posix())
    return path


def save_final_model(model: Any, paths: TrainingPaths) -> Path:
    """Save the final model snapshot."""

    path = paths.models_dir / "final_model.zip"
    model.save(path.as_posix())
    return path


def save_evaluation_report(
    paths: TrainingPaths,
    *,
    timesteps: int,
    stage_name: str,
    report: dict[str, object],
) -> Path:
    """Persist one evaluation report as JSON."""

    path = paths.evaluations_dir / f"eval_step_{timesteps:09d}.json"
    payload = {
        "timesteps": timesteps,
        "stage_name": stage_name,
        "report": report,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_run_manifest(paths: TrainingPaths, payload: dict[str, object]) -> Path:
    """Persist top-level run metadata as JSON."""

    path = paths.run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def load_sac_model(model_path: Path, *, env: Any | None = None, device: str = "auto") -> Any:
    """Load a SAC model from disk."""

    from stable_baselines3 import SAC

    return SAC.load(model_path.as_posix(), env=env, device=device)
