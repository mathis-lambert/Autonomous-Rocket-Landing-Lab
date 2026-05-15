"""Configuration objects for SB3-based training."""

from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.application.rl.config import EvaluationConfig


@dataclass(frozen=True, slots=True)
class SACHyperparameters:
    """Core SAC hyperparameters for the first training setup."""

    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    buffer_size: int = 500_000
    learning_starts: int = 2_000
    batch_size: int = 256
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpointing policy for the trainer."""

    save_freq_timesteps: int = 25_000
    save_best_model: bool = True
    save_latest_model: bool = True


@dataclass(frozen=True, slots=True)
class SB3TrainerConfig:
    """Top-level training configuration."""

    total_timesteps: int = 250_000
    train_segment_timesteps: int = 10_000
    n_envs: int = 1
    seed: int | None = None
    verbose: int = 1
    progress_bar: bool = False
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    sac: SACHyperparameters = field(default_factory=SACHyperparameters)
