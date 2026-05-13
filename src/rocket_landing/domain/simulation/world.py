"""Stateful world wrapper around the lower-level simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State
from rocket_landing.domain.simulation.engine import SimulationEngine


@dataclass(slots=True)
class SimulationWorld:
    """Own the mutable world state and the fixed simulation step size."""

    params: RocketParams
    dt: float
    state: State
    engine: SimulationEngine = field(init=False)
    time: float = 0.0

    def __post_init__(self) -> None:
        """Build the engine lazily once the world has its parameter set."""

        self.engine = SimulationEngine(self.params)

    def step(self, action: Action) -> StepResult:
        """Advance the world by one fixed time step."""

        result = self.engine.step(self.state, action, self.dt)
        self.state = result.state
        self.time += self.dt
        return result
