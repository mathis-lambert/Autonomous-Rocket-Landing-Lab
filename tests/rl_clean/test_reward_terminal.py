from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State
from rocket_landing.rl.reward import build_episode_summary, compute_reward
from rocket_landing.rl.types import RewardConfig, RewardWeights


def test_truncation_has_terminal_penalty() -> None:
    params = RocketParams()
    state = State(
        x=0.0,
        z=10.0,
        vx=0.0,
        vz=-1.0,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )
    reward = compute_reward(
        action=Action(throttle=0.0, engine_gimbal=0.0),
        previous_state=state,
        result=StepResult(state=state, terminated=False, landed=False, crashed=False),
        params=params,
        config=RewardConfig(truncation_penalty=123.0),
        truncated=True,
    )

    assert reward.terminal == -123.0


def test_near_ground_penalty_targets_landing_constraint_excess() -> None:
    params = RocketParams()
    previous_state = State(
        x=0.0,
        z=40.0,
        vx=0.0,
        vz=-1.0,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )
    high_state = State(
        x=0.0,
        z=40.0,
        vx=0.0,
        vz=-5.0,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )
    low_state = State(
        x=0.0,
        z=params.length * 0.5,
        vx=0.0,
        vz=-5.0,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )
    config = RewardConfig(
        weights=RewardWeights(vz=0.0, near_ground_vz=1.0),
        near_ground_altitude=20.0,
    )

    high_reward = compute_reward(
        action=Action(throttle=0.0, engine_gimbal=0.0),
        previous_state=previous_state,
        result=StepResult(state=high_state, terminated=False, landed=False, crashed=False),
        params=params,
        config=config,
    )
    low_reward = compute_reward(
        action=Action(throttle=0.0, engine_gimbal=0.0),
        previous_state=previous_state,
        result=StepResult(state=low_state, terminated=False, landed=False, crashed=False),
        params=params,
        config=config,
    )
    safe_low_state = State(
        x=0.0,
        z=params.length * 0.5,
        vx=0.0,
        vz=-params.max_landing_vz,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )
    safe_low_reward = compute_reward(
        action=Action(throttle=0.0, engine_gimbal=0.0),
        previous_state=previous_state,
        result=StepResult(state=safe_low_state, terminated=False, landed=False, crashed=False),
        params=params,
        config=config,
    )

    assert high_reward.near_ground_vz == 0.0
    assert safe_low_reward.near_ground_vz == 0.0
    assert low_reward.near_ground_vz == -(5.0 - params.max_landing_vz)


def test_guidance_reward_targets_lateral_progress_at_altitude() -> None:
    params = RocketParams()
    previous_state = State(
        x=10.0,
        z=120.0,
        vx=1.0,
        vz=-8.0,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )
    state = State(
        x=7.0,
        z=118.0,
        vx=2.0,
        vz=-8.0,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )
    config = RewardConfig(
        weights=RewardWeights(guidance_dx_progress=2.0, guidance_vx=0.5),
        guidance_altitude=50.0,
    )

    reward = compute_reward(
        action=Action(throttle=0.0, engine_gimbal=0.0),
        previous_state=previous_state,
        result=StepResult(state=state, terminated=False, landed=False, crashed=False),
        params=params,
        config=config,
    )

    assert reward.guidance_dx_progress == 6.0
    assert reward.guidance_vx == -1.0


def test_episode_summary_exposes_landing_constraint_components() -> None:
    params = RocketParams()
    impact_state = State(
        x=1.0,
        z=params.length * 0.5,
        vx=1.2,
        vz=-2.5,
        theta=0.03,
        omega=0.04,
        fuel=params.initial_fuel,
    )
    summary = build_episode_summary(
        step_count=10,
        elapsed_time=0.2,
        episode_return=12.0,
        result=StepResult(
            state=impact_state,
            terminated=True,
            landed=True,
            crashed=False,
            impact_state=impact_state,
        ),
        params=params,
        truncated=False,
    )

    assert summary["truncation_reason"] is None
    assert summary["final_vx"] == impact_state.vx
    assert summary["final_vz"] == impact_state.vz
    assert summary["final_theta"] == impact_state.theta
    assert summary["final_omega"] == impact_state.omega
