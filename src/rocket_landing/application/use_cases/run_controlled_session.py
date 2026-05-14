"""Use cases for live, controller-driven simulation sessions.

This module exposes a mutable session object that is convenient for interactive
clients.  It is intentionally stateful, unlike lower-level domain components
that remain as pure and focused as possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.domain.simulation.world import SimulationWorld


@dataclass(slots=True)
class ControlledSimulationSession:
    """Mutable simulation session designed for live control loops.

    Responsibilities:
    - own the current ``SimulationWorld``
    - expose read-only accessors used by UI code
    - append every step to a ``SimulationHistory``
    - report high-level termination status to clients
    """

    params: RocketParams
    dt: float
    initial_state: State
    max_steps: int = 50_000
    _world: SimulationWorld = field(init=False, repr=False)
    _history: SimulationHistory = field(init=False, repr=False)
    _step_count: int = field(init=False, default=0)
    _last_result: StepResult | None = field(init=False, default=None)
    _default_initial_state: State = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize the internal world and trajectory buffers."""

        self._default_initial_state = self.initial_state
        self.reset()

    @property
    def state(self) -> State:
        """Expose the current world state."""

        return self._world.state

    @property
    def time(self) -> float:
        """Expose the elapsed simulation time in seconds."""

        return self._world.time

    @property
    def history(self) -> SimulationHistory:
        """Expose the accumulated trajectory history."""

        return self._history

    @property
    def step_count(self) -> int:
        """Return the number of applied control steps."""

        return self._step_count

    @property
    def current_frame_index(self) -> int:
        """Return the history index corresponding to the current state."""

        return len(self._history.states) - 1

    @property
    def latest_action(self) -> Action:
        """Return the most recently recorded action sample."""

        return self._history.actions[-1]

    @property
    def is_finished(self) -> bool:
        """Return whether the session is terminated or exhausted."""

        if self._last_result is not None and self._last_result.terminated:
            return True
        return self._step_count >= self.max_steps

    @property
    def last_result(self) -> StepResult | None:
        """Return the last step result if the session has advanced."""

        return self._last_result

    def reset(self, initial_state: State | None = None) -> None:
        """Reset the session to a provided or default initial state.

        Args:
            initial_state: Optional explicit spawn state. When omitted, the
                session resets to the scenario state provided at construction.
        """

        start_state = initial_state or self._default_initial_state
        self._world = SimulationWorld(params=self.params, dt=self.dt, state=start_state)
        self._history = SimulationHistory()
        self._history.append(
            time=self._world.time,
            state=self._world.state,
            action=Action.neutral(),
        )
        self._step_count = 0
        self._last_result = None

    def step(self, action: Action) -> StepResult:
        """Advance the session and append the new sample to history.

        Args:
            action: Control command to apply for the next fixed time step.

        Returns:
            The resolved result produced by the simulation world.
        """

        result = self._world.step(action)
        self._step_count += 1
        self._history.append(time=self._world.time, state=self._world.state, action=action)
        self._last_result = result
        return result
