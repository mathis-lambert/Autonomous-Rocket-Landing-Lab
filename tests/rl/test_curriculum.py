from rocket_landing.application.rl.curriculum.scheduler import CurriculumScheduler
from rocket_landing.application.rl.curriculum.stages import CurriculumStage, PromotionCriteria
from rocket_landing.application.rl.evaluation.metrics import EvaluationMetrics
from rocket_landing.application.rl.reset.distributions import InitialStateDistribution, UniformRange
from rocket_landing.application.rl.termination import TruncationConfig


def _stage(name: str, min_success_rate: float) -> CurriculumStage:
    return CurriculumStage(
        name=name,
        reset_distribution=InitialStateDistribution(
            dx=UniformRange(0.0, 0.0),
            z=UniformRange(100.0, 100.0),
            vx=UniformRange(0.0, 0.0),
            vz=UniformRange(-10.0, -10.0),
            theta=UniformRange(0.0, 0.0),
            omega=UniformRange(0.0, 0.0),
            fuel_ratio=UniformRange(1.0, 1.0),
        ),
        truncation=TruncationConfig(),
        promotion=PromotionCriteria(min_success_rate=min_success_rate, min_eval_episodes=5),
    )


def test_curriculum_scheduler_advances_when_metrics_meet_threshold() -> None:
    scheduler = CurriculumScheduler((_stage("easy", 0.6), _stage("hard", 0.8)))
    metrics = EvaluationMetrics(
        episodes=5,
        success_rate=0.7,
        crash_rate=0.2,
        truncation_rate=0.1,
        mean_return=10.0,
        mean_final_dx=1.0,
        mean_abs_final_dx=1.0,
        mean_final_altitude=1.0,
        mean_final_speed=1.0,
        mean_fuel_ratio=0.5,
    )

    advanced = scheduler.maybe_advance(metrics)

    assert advanced is True
    assert scheduler.current_stage.name == "hard"


def test_curriculum_scheduler_stops_advancing_on_final_stage() -> None:
    scheduler = CurriculumScheduler((_stage("final", 0.6),))
    metrics = EvaluationMetrics(
        episodes=10,
        success_rate=1.0,
        crash_rate=0.0,
        truncation_rate=0.0,
        mean_return=50.0,
        mean_final_dx=0.0,
        mean_abs_final_dx=0.0,
        mean_final_altitude=0.0,
        mean_final_speed=0.0,
        mean_fuel_ratio=0.9,
    )

    advanced = scheduler.maybe_advance(metrics)

    assert advanced is False
    assert scheduler.current_stage.name == "final"
