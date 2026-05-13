"""Use case for running non-interactive, constant-command simulations."""

from __future__ import annotations

from dataclasses import dataclass

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State
from rocket_landing.domain.simulation.history import SimulationHistory
from rocket_landing.domain.simulation.world import SimulationWorld


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """Immutable bundle returned after a scripted simulation run."""

    history: SimulationHistory
    final_result: StepResult
    final_time: float


class RunConstantAction:
    """Use case for simulating a fixed command over time."""

    def __init__(self, params: RocketParams, dt: float, *, initial_state: State) -> None:
        self._params = params
        self._dt = dt
        self._initial_state = initial_state

    def execute(self, *, action: Action, max_steps: int) -> SimulationRun:
        """Run the world forward until termination or the step budget is reached."""

        world = SimulationWorld(
            params=self._params,
            dt=self._dt,
            state=self._initial_state,
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
