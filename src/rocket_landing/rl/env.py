"""Gymnasium environment for rocket landing RL."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np

from rocket_landing.application.use_cases.run_controlled_session import ControlledSimulationSession
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State
from rocket_landing.rl.reset import build_nominal_initial_state, sample_initial_state
from rocket_landing.rl.reward import RewardBreakdown, build_episode_summary, compute_reward
from rocket_landing.rl.types import (
    CurriculumStage,
    EnvConfig,
    RewardConfig,
    TruncationConfig,
)

ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float32)
ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)
OBSERVATION_SCALE = np.array([250.0, 500.0, 50.0, 50.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32)
OBSERVATION_LOW = np.array(
    [-1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32
)
OBSERVATION_HIGH = np.array(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
)


def fuel_ratio_for(state: State, params: RocketParams) -> float:
    """Return normalized remaining propellant."""

    if params.initial_fuel <= 0.0:
        return 0.0
    return min(max(state.fuel / params.initial_fuel, 0.0), 1.0)


def build_observation(state: State, params: RocketParams) -> np.ndarray:
    """Encode domain state into a compact observation vector."""

    raw = np.array(
        [
            state.horizontal_error(params.target_x),
            state.z,
            state.vx,
            state.vz,
            math.sin(state.theta),
            math.cos(state.theta),
            state.omega,
            fuel_ratio_for(state, params),
        ],
        dtype=np.float32,
    )
    return raw / OBSERVATION_SCALE


def decode_action(raw_action: np.ndarray, params: RocketParams) -> Action:
    """Map symmetric policy output [-1, 1] to domain control commands."""

    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.shape != (2,):
        raise ValueError(f"expected action shape (2,), got {action.shape}")
    throttle = float(0.5 * (np.clip(action[0], -1.0, 1.0) + 1.0))
    gimbal = float(np.clip(action[1], -1.0, 1.0) * params.max_gimbal)
    return Action(throttle=throttle, engine_gimbal=gimbal, aero_steer=0.0)


def truncation_reason_for(
    *,
    state: State,
    step_count: int,
    params: RocketParams,
    config: TruncationConfig,
    max_episode_steps: int,
) -> str | None:
    """Return the training-only stop reason, if the episode should truncate."""

    if step_count >= max_episode_steps:
        return "max_episode_steps"
    if abs(state.horizontal_error(params.target_x)) > config.max_abs_dx:
        return "max_abs_dx"
    if state.z > config.max_altitude:
        return "max_altitude"
    return None


class RocketLanderEnv(gym.Env):
    """Continuous-control environment around the rocket simulator."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: RocketParams,
        *,
        env_config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        default_initial_state: State | None = None,
    ) -> None:
        super().__init__()
        self._params = params
        self._env_config = env_config or EnvConfig()
        self._base_reward_config = reward_config or RewardConfig()
        self._reward_config = self._base_reward_config
        self._reset_distribution = self._env_config.reset_distribution
        self._truncation_config = self._env_config.truncation
        self._default_initial_state = default_initial_state or build_nominal_initial_state(params)
        self._session = ControlledSimulationSession(
            params=params,
            dt=self._env_config.dt,
            initial_state=self._default_initial_state,
            max_steps=self._env_config.max_episode_steps,
        )
        self._episode_return = 0.0
        self._needs_reset = True
        self._stage_name = "default"

        self.action_space = gym.spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=OBSERVATION_LOW,
            high=OBSERVATION_HIGH,
            dtype=np.float32,
        )

    @property
    def params(self) -> RocketParams:
        return self._params

    @property
    def session(self) -> ControlledSimulationSession:
        return self._session

    @property
    def current_stage_name(self) -> str:
        return self._stage_name

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        start_state = self._resolve_reset_state(options)
        self._session.reset(start_state)
        self._episode_return = 0.0
        self._needs_reset = False
        return build_observation(self._session.state, self._params), {
            "dx": self._session.state.horizontal_error(self._params.target_x),
            "fuel_ratio": fuel_ratio_for(self._session.state, self._params),
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if self._needs_reset:
            raise RuntimeError("environment must be reset before calling step()")
        previous_state = self._session.state
        domain_action = decode_action(action, self._params)
        result = self._session.step(domain_action)

        terminated = result.terminated
        truncation_reason = None
        if not terminated:
            truncation_reason = truncation_reason_for(
                state=result.state,
                step_count=self._session.step_count,
                params=self._params,
                config=self._truncation_config,
                max_episode_steps=self._env_config.max_episode_steps,
            )
        truncated = truncation_reason is not None
        reward = compute_reward(
            action=domain_action,
            previous_state=previous_state,
            result=result,
            params=self._params,
            config=self._reward_config,
            truncated=truncated,
        )
        self._episode_return += reward.total
        if terminated or truncated:
            self._needs_reset = True

        observation = build_observation(self._session.state, self._params)
        info = self._build_step_info(
            result=result,
            reward=reward,
            truncated=truncated,
            truncation_reason=truncation_reason,
        )
        return observation, reward.total, terminated, truncated, info

    def close(self) -> None:
        """Release environment resources."""

    def apply_curriculum_stage(self, stage: CurriculumStage) -> None:
        """Apply one curriculum stage to this environment."""

        self._stage_name = stage.name
        self._reset_distribution = stage.reset_distribution
        self._truncation_config = stage.truncation
        self._reward_config = stage.reward_config or self._base_reward_config

    def _resolve_reset_state(self, options: dict[str, object] | None) -> State:
        if options is not None:
            explicit_state = options.get("initial_state")
            if isinstance(explicit_state, State):
                return explicit_state
        return sample_initial_state(self._params, self._reset_distribution, self.np_random)

    def _build_step_info(
        self,
        *,
        result: StepResult,
        reward: RewardBreakdown,
        truncated: bool,
        truncation_reason: str | None,
    ) -> dict[str, object]:
        state = self._session.state
        info: dict[str, object] = {
            "dx": state.horizontal_error(self._params.target_x),
            "fuel_ratio": fuel_ratio_for(state, self._params),
            "landed": result.landed,
            "crashed": result.crashed,
            "truncation_reason": truncation_reason,
            "curriculum_stage": self._stage_name,
            "reward": reward.as_dict(),
        }
        if result.terminated or truncated:
            info["episode"] = build_episode_summary(
                step_count=self._session.step_count,
                elapsed_time=self._session.time,
                episode_return=self._episode_return,
                result=result,
                params=self._params,
                truncated=truncated,
                truncation_reason=truncation_reason,
            )
        return info
