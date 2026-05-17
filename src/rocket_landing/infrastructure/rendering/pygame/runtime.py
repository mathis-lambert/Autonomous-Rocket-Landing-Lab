"""Runtime adapters that let the pygame app drive different simulation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rocket_landing.application.use_cases.run_controlled_session import ControlledSimulationSession
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.rl.env import RocketLanderEnv


class SimulationRuntime(Protocol):
    """Common runtime surface required by the generic pygame app."""

    @property
    def state(self) -> State: ...

    @property
    def time(self) -> float: ...

    @property
    def history(self) -> SimulationHistory: ...

    @property
    def step_count(self) -> int: ...

    @property
    def current_frame_index(self) -> int: ...

    @property
    def latest_action(self) -> Action: ...

    @property
    def is_finished(self) -> bool: ...

    @property
    def last_result(self) -> StepResult | None: ...

    @property
    def max_steps(self) -> int: ...

    def reset(self) -> None: ...

    def step(self, action: Action) -> None: ...


@dataclass(slots=True)
class SessionRuntime:
    """Adapter around the manual controlled simulation session."""

    session: ControlledSimulationSession

    @property
    def state(self) -> State:
        return self.session.state

    @property
    def time(self) -> float:
        return self.session.time

    @property
    def history(self) -> SimulationHistory:
        return self.session.history

    @property
    def step_count(self) -> int:
        return self.session.step_count

    @property
    def current_frame_index(self) -> int:
        return self.session.current_frame_index

    @property
    def latest_action(self) -> Action:
        return self.session.latest_action

    @property
    def is_finished(self) -> bool:
        return self.session.is_finished

    @property
    def last_result(self) -> StepResult | None:
        return self.session.last_result

    @property
    def max_steps(self) -> int:
        return self.session.max_steps

    def reset(self) -> None:
        self.session.reset()

    def step(self, action: Action) -> None:
        self.session.step(action)


@dataclass(slots=True)
class EnvironmentRuntime:
    """Adapter around the RL environment for policy playback."""

    env: RocketLanderEnv

    @property
    def state(self) -> State:
        return self.env.session.state

    @property
    def time(self) -> float:
        return self.env.session.time

    @property
    def history(self) -> SimulationHistory:
        return self.env.session.history

    @property
    def step_count(self) -> int:
        return self.env.session.step_count

    @property
    def current_frame_index(self) -> int:
        return self.env.session.current_frame_index

    @property
    def latest_action(self) -> Action:
        return self.env.session.latest_action

    @property
    def is_finished(self) -> bool:
        return self.env.episode_finished

    @property
    def last_result(self) -> StepResult | None:
        return self.env.session.last_result

    @property
    def max_steps(self) -> int:
        return self.env.session.max_steps

    def reset(self) -> None:
        self.env.reset()

    def step(self, action: Action) -> None:
        self.env.step_domain_action(action)
