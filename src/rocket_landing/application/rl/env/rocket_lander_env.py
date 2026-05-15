"""Gymnasium environment adapter around the rocket landing simulator."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from rocket_landing.application.rl.config.env_config import RLEnvConfig
from rocket_landing.application.rl.config.reward_config import RewardConfig
from rocket_landing.application.rl.curriculum.stages import CurriculumStage
from rocket_landing.application.rl.encoding.actions import decode_agent_action
from rocket_landing.application.rl.encoding.observations import build_observation, fuel_ratio_for
from rocket_landing.application.rl.env.env_info import build_reset_info, build_step_info
from rocket_landing.application.rl.reset.distributions import InitialStateDistribution
from rocket_landing.application.rl.reset.initial_state_sampler import (
    build_nominal_initial_state,
    sample_initial_state,
)
from rocket_landing.application.rl.reward.reward_function import compute_reward
from rocket_landing.application.rl.specs.action_spec import ACTION_HIGH, ACTION_LOW
from rocket_landing.application.rl.specs.episode_types import EpisodeSummary
from rocket_landing.application.rl.specs.observation_spec import OBSERVATION_HIGH, OBSERVATION_LOW
from rocket_landing.application.rl.termination.termination_rules import resolve_step_flags
from rocket_landing.application.rl.termination.truncation_rules import TruncationConfig
from rocket_landing.application.use_cases.run_controlled_session import ControlledSimulationSession
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State


class RocketLanderEnv(gym.Env):
    """Minimal continuous-control environment for SAC-style agents."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: RocketParams,
        *,
        env_config: RLEnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        default_initial_state: State | None = None,
    ) -> None:
        super().__init__()
        self._params = params
        self._env_config = env_config or RLEnvConfig()
        self._reward_config = reward_config or RewardConfig()
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
        self._curriculum_stage_name = "default"

        self.action_space = gym.spaces.Box(low=ACTION_LOW, high=ACTION_HIGH, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=OBSERVATION_LOW,
            high=OBSERVATION_HIGH,
            dtype=np.float32,
        )

    @property
    def params(self) -> RocketParams:
        """Expose the immutable parameter set used by the environment."""

        return self._params

    @property
    def session(self) -> ControlledSimulationSession:
        """Expose the underlying controlled session for playback and inspection."""

        return self._session

    @property
    def reward_config(self) -> RewardConfig:
        """Expose the active reward configuration."""

        return self._reward_config

    @property
    def current_stage_name(self) -> str:
        """Return the human-readable name of the active curriculum stage."""

        return self._curriculum_stage_name

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Reset the simulator and return the first observation."""

        super().reset(seed=seed)
        start_state = self._resolve_reset_state(options)
        self._session.reset(start_state)
        self._episode_return = 0.0
        self._needs_reset = False
        return build_observation(self._session.state, self._params), build_reset_info(
            self._session.state,
            self._params,
        )

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        """Advance the simulation by one fixed control step."""

        if self._needs_reset:
            raise RuntimeError("environment must be reset before calling step()")

        previous_state = self._session.state
        domain_action = decode_agent_action(action, self._params)
        result = self._session.step(domain_action)
        reward = compute_reward(
            action=domain_action,
            previous_state=previous_state,
            result=result,
            params=self._params,
            config=self._reward_config,
        )
        self._episode_return += reward.total

        flags = resolve_step_flags(
            result=result,
            step_count=self._session.step_count,
            params=self._params,
            config=self._truncation_config,
            max_episode_steps=self._env_config.max_episode_steps,
        )
        if flags.terminated or flags.truncated:
            self._needs_reset = True

        observation = build_observation(self._session.state, self._params)
        episode_summary = None
        if flags.terminated or flags.truncated:
            episode_summary = self._build_episode_summary(result, flags.truncated)
        info = build_step_info(
            state=self._session.state,
            params=self._params,
            result=result,
            reward=reward,
            curriculum_stage=self._curriculum_stage_name,
            episode_summary=episode_summary,
        )
        return observation, reward.total, flags.terminated, flags.truncated, info

    def render(self) -> None:
        """Rendering is delegated to the existing pygame infrastructure."""

        raise NotImplementedError("use the pygame live app for visualization")

    def close(self) -> None:
        """Release environment resources."""

    def set_reset_distribution(self, distribution: InitialStateDistribution) -> None:
        """Replace the active reset sampler distribution."""

        self._reset_distribution = distribution

    def set_truncation_config(self, config: TruncationConfig) -> None:
        """Replace the active truncation rules."""

        self._truncation_config = config

    def set_reward_config(self, config: RewardConfig) -> None:
        """Replace the active reward shaping parameters."""

        self._reward_config = config

    def apply_curriculum_stage(self, stage: CurriculumStage) -> None:
        """Apply one curriculum stage worth of runtime overrides."""

        self._curriculum_stage_name = stage.name
        self.set_reset_distribution(stage.reset_distribution)
        self.set_truncation_config(stage.truncation)
        if stage.reward_config is not None:
            self.set_reward_config(stage.reward_config)

    def _resolve_reset_state(self, options: dict[str, object] | None) -> State:
        """Resolve a reset state from explicit options or from the sampler."""

        if options is not None:
            explicit_state = options.get("initial_state")
            if isinstance(explicit_state, State):
                return explicit_state
        return sample_initial_state(
            self._params,
            self._reset_distribution,
            self.np_random,
        )

    def _build_episode_summary(
        self,
        result: StepResult,
        truncated: bool,
    ) -> EpisodeSummary:
        """Build compact end-of-episode telemetry."""

        state = result.impact_state or result.state
        return EpisodeSummary(
            steps=self._session.step_count,
            elapsed_time=self._session.time,
            return_total=self._episode_return,
            landed=result.landed,
            crashed=result.crashed,
            truncated=truncated,
            final_dx=state.horizontal_error(self._params.target_x),
            final_altitude=state.z,
            final_speed=state.speed,
            fuel_ratio=fuel_ratio_for(state, self._params),
        )
