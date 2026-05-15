"""Build structured ``info`` payloads for the RL environment."""

from __future__ import annotations

from rocket_landing.application.rl.encoding.observations import fuel_ratio_for
from rocket_landing.application.rl.reward.reward_breakdown import RewardBreakdown
from rocket_landing.application.rl.specs.episode_types import EpisodeSummary
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State


def build_reset_info(state: State, params: RocketParams) -> dict[str, float]:
    """Return reset metadata that is useful for debugging rollouts."""

    return {
        "dx": state.horizontal_error(params.target_x),
        "fuel_ratio": fuel_ratio_for(state, params),
    }


def build_step_info(
    *,
    state: State,
    params: RocketParams,
    result: StepResult,
    reward: RewardBreakdown,
    curriculum_stage: str,
    episode_summary: EpisodeSummary | None,
) -> dict[str, object]:
    """Return step-level metadata aligned with Gymnasium conventions."""

    info: dict[str, object] = {
        "dx": state.horizontal_error(params.target_x),
        "fuel_ratio": fuel_ratio_for(state, params),
        "landed": result.landed,
        "crashed": result.crashed,
        "curriculum_stage": curriculum_stage,
        "reward": reward.as_dict(),
    }
    if episode_summary is not None:
        info["episode"] = episode_summary.as_dict()
    return info
