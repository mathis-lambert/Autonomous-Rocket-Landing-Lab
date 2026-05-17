from rocket_landing.domain.models.params import RocketParams
from rocket_landing.rl import TrainingConfig, build_default_curriculum
from rocket_landing.rl.campaign import (
    apply_training_overrides,
    parse_seed_list,
    slice_curriculum_from_stage,
)


def test_parse_seed_list_rejects_empty_input() -> None:
    try:
        parse_seed_list(" , ")
    except ValueError as exc:
        assert "at least one seed" in str(exc)
    else:
        raise AssertionError("expected parse_seed_list to reject empty seed input")


def test_slice_curriculum_from_stage_keeps_suffix_order() -> None:
    stages = build_default_curriculum(RocketParams())

    sliced = slice_curriculum_from_stage("wide_recovery", stages)

    assert [stage.name for stage in sliced] == [
        "wide_recovery",
        "full_envelope_nominal_fuel",
        "full_envelope",
    ]


def test_apply_training_overrides_updates_only_selected_sac_fields() -> None:
    config = apply_training_overrides(
        TrainingConfig(),
        learning_rate=1e-4,
        batch_size=128,
    )

    assert config.sac.learning_rate == 1e-4
    assert config.sac.batch_size == 128
    assert config.sac.gradient_steps == 1
