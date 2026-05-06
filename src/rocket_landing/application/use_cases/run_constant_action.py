from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.domain.simulation.world import SimulationWorld, default_initial_state


@dataclass(frozen=True, slots=True)
class SimulationRun:
    history: SimulationHistory
    final_result: StepResult
    final_time: float


class RunConstantAction:
    """Use case for simulating a fixed command over time."""

    def __init__(self, params: RocketParams, dt: float) -> None:
        self._params = params
        self._dt = dt

    def execute(self, *, action: Action, max_steps: int) -> SimulationRun:
        world = SimulationWorld(
            params=self._params,
            dt=self._dt,
            state=default_initial_state(self._params),
        )
        history = SimulationHistory()
        history.append(time=world.time, state=world.state, action=action)

        result = StepResult(state=world.state, terminated=False, landed=False, crashed=False)
        for _ in range(max_steps):
            result = world.step(action)
            history.append(time=world.time, state=world.state, action=action)
            if result.terminated:
                break

        return SimulationRun(history=history, final_result=result, final_time=world.time)
