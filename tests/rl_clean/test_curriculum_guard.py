from dataclasses import dataclass

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.rl.curriculum import (
    CurriculumScheduler,
    build_default_curriculum,
    meets_promotion_criteria,
)


@dataclass(frozen=True, slots=True)
class Metrics:
    episodes: int
    success_rate: float
    mean_abs_final_dx: float
    crash_rate: float = 0.0
    truncation_rate: float = 0.0
    mean_abs_final_vx: float = 0.0
    mean_abs_final_vz: float = 0.0
    mean_abs_final_theta: float = 0.0
    mean_abs_final_omega: float = 0.0


def test_scheduler_exposes_next_stage_without_advancing() -> None:
    scheduler = CurriculumScheduler(build_default_curriculum(RocketParams()))

    assert scheduler.current_stage.name == "touchdown"
    assert scheduler.next_stage is not None
    assert scheduler.next_stage.name == "stabilize"
    assert scheduler.current_stage.name == "touchdown"


def test_meets_promotion_criteria_checks_success_and_lateral_error() -> None:
    stage = build_default_curriculum(RocketParams())[0]

    assert meets_promotion_criteria(stage, Metrics(50, 0.85, 1.0))
    assert not meets_promotion_criteria(stage, Metrics(49, 1.0, 0.0))
    assert not meets_promotion_criteria(stage, Metrics(50, 0.84, 0.0))
    assert not meets_promotion_criteria(stage, Metrics(50, 1.0, 10.0))
    assert not meets_promotion_criteria(stage, Metrics(50, 1.0, 0.0, crash_rate=0.2))
    assert not meets_promotion_criteria(stage, Metrics(50, 1.0, 0.0, truncation_rate=0.2))
    assert not meets_promotion_criteria(stage, Metrics(50, 1.0, 0.0, mean_abs_final_vx=1.2))


def test_curriculum_adds_bridge_stages_before_full_envelope() -> None:
    stages = build_default_curriculum(RocketParams())
    names = [stage.name for stage in stages]

    assert names[-3:] == ["wide_recovery", "full_envelope_nominal_fuel", "full_envelope"]
