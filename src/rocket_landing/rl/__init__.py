"""Public API for the rebuilt RL stack."""

from rocket_landing.rl.curriculum import (
    CurriculumScheduler,
    build_default_curriculum,
    find_stage_by_name,
)
from rocket_landing.rl.env import RocketLanderEnv
from rocket_landing.rl.evaluation import EvaluationReport, evaluate_policy
from rocket_landing.rl.sb3 import TrainingSummary, load_sac_model, train_sac
from rocket_landing.rl.types import (
    CurriculumGuardConfig,
    CurriculumStage,
    EnvConfig,
    EvaluationConfig,
    RewardConfig,
    TrainingConfig,
)

__all__ = [
    "CurriculumScheduler",
    "CurriculumGuardConfig",
    "CurriculumStage",
    "EnvConfig",
    "EvaluationConfig",
    "EvaluationReport",
    "RewardConfig",
    "RocketLanderEnv",
    "TrainingConfig",
    "TrainingSummary",
    "build_default_curriculum",
    "evaluate_policy",
    "find_stage_by_name",
    "load_sac_model",
    "train_sac",
]
