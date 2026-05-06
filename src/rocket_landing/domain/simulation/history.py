from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.state import State


@dataclass(slots=True)
class SimulationHistory:
    times: list[float] = field(default_factory=list)
    states: list[State] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)

    def append(self, *, time: float, state: State, action: Action) -> None:
        self.times.append(time)
        self.states.append(state)
        self.actions.append(action)

    def is_empty(self) -> bool:
        return not self.states
