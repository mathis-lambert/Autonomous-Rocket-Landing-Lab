"""Core types and configuration objects for the RL stack."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class UniformRange:
    """Closed range used by the reset sampler."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError("uniform range high bound must be >= low bound")


@dataclass(frozen=True, slots=True)
class InitialStateDistribution:
    """Independent ranges used to sample a reset state."""

    dx: UniformRange
    z: UniformRange
    vx: UniformRange
    vz: UniformRange
    theta: UniformRange
    omega: UniformRange
    fuel_ratio: UniformRange


def default_initial_state_distribution() -> InitialStateDistribution:
    """Return the full-envelope reset distribution."""

    return InitialStateDistribution(
        dx=UniformRange(-25.0, 25.0),
        z=UniformRange(80.0, 140.0),
        vx=UniformRange(-6.0, 6.0),
        vz=UniformRange(-20.0, -8.0),
        theta=UniformRange(-0.10, 0.10),
        omega=UniformRange(-0.08, 0.08),
        fuel_ratio=UniformRange(0.55, 1.0),
    )


@dataclass(frozen=True, slots=True)
class RewardWeights:
    """Dense shaping weights applied each step."""

    dx: float = 0.12
    z: float = 0.08
    vx: float = 0.12
    vz: float = 0.15
    tilt: float = 0.20
    omega: float = 0.06
    throttle: float = 0.005
    gimbal: float = 0.002
    step: float = 0.15
    dx_progress: float = 0.5
    z_progress: float = 1.0
    guidance_dx_progress: float = 0.0
    guidance_vx: float = 0.0
    near_ground_vx: float = 0.25
    near_ground_vz: float = 0.45
    near_ground_tilt: float = 0.80
    near_ground_omega: float = 0.25


@dataclass(frozen=True, slots=True)
class RewardConfig:
    """Terminal and dense reward configuration."""

    weights: RewardWeights = field(default_factory=RewardWeights)
    landing_bonus: float = 2_000.0
    crash_penalty: float = 800.0
    truncation_penalty: float = 600.0
    guidance_altitude: float = 40.0
    near_ground_altitude: float = 6.0


@dataclass(frozen=True, slots=True)
class TruncationConfig:
    """Hard rollout bounds used for early truncation."""

    max_abs_dx: float = 250.0
    max_altitude: float = 500.0


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Environment runtime configuration."""

    dt: float = 0.02
    max_episode_steps: int = 1_000
    truncation: TruncationConfig = field(default_factory=TruncationConfig)
    reset_distribution: InitialStateDistribution = field(
        default_factory=default_initial_state_distribution
    )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Policy evaluation configuration."""

    episodes: int = 50
    deterministic: bool = True
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class PromotionCriteria:
    """Conditions needed before moving to next curriculum stage."""

    min_success_rate: float
    min_eval_episodes: int = 50
    max_mean_abs_final_dx: float | None = None
    max_crash_rate: float | None = None
    max_truncation_rate: float | None = None
    max_mean_abs_final_vx: float | None = None
    max_mean_abs_final_vz: float | None = None
    max_mean_abs_final_theta: float | None = None
    max_mean_abs_final_omega: float | None = None


@dataclass(frozen=True, slots=True)
class CurriculumGuardConfig:
    """Safety checks that prevent curriculum training from degrading a policy."""

    enable_next_stage_probe: bool = True
    rollback_success_drop: float = 0.35
    rollback_min_best_success_rate: float = 0.50
    max_rollbacks_per_stage: int = 2


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """Runtime overrides grouped as one curriculum stage."""

    name: str
    reset_distribution: InitialStateDistribution
    truncation: TruncationConfig
    promotion: PromotionCriteria
    reward_config: RewardConfig | None = None


@dataclass(frozen=True, slots=True)
class SACHyperparameters:
    """SAC hyperparameters used by the trainer."""

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
    """Checkpointing policy."""

    save_freq_timesteps: int = 25_000
    save_best_model: bool = True
    save_latest_model: bool = True


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Top-level trainer configuration."""

    total_timesteps: int = 250_000
    segment_timesteps: int = 10_000
    n_envs: int = 1
    seed: int | None = None
    verbose: int = 1
    progress_bar: bool = False
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    sac: SACHyperparameters = field(default_factory=SACHyperparameters)
    curriculum_guard: CurriculumGuardConfig = field(default_factory=CurriculumGuardConfig)
