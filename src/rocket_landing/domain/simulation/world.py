from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State
from rocket_landing.domain.simulation.engine import SimulationEngine


@dataclass(slots=True)
class SimulationWorld:
    params: RocketParams
    dt: float
    state: State
    engine: SimulationEngine = field(init=False)
    time: float = 0.0

    def __post_init__(self) -> None:
        self.engine = SimulationEngine(self.params)

    def step(self, action: Action) -> StepResult:
        result = self.engine.step(self.state, action, self.dt)
        self.state = result.state
        self.time += self.dt
        return result


def default_initial_state(params: RocketParams) -> State:
    return State(
        x=0.0,
        z=120.0,
        vx=0.0,
        vz=-15.0,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )
