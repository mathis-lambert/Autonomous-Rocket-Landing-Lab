"""Ground contact rules for deciding landings and crashes."""

from __future__ import annotations

from dataclasses import replace

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.results import StepResult
from rocket_landing.domain.models.state import State


class GroundContactResolver:
    """Determines whether a ground contact is safe or a crash."""

    def __init__(self, params: RocketParams) -> None:
        self._params = params

    def resolve(self, state: State) -> StepResult:
        """Snap the booster to the ground and classify the contact outcome."""

        if state.z > 0.0:
            return StepResult(state=state, terminated=False, landed=False, crashed=False)

        grounded_state = replace(state, z=0.0)
        landed = self._is_soft_landing(grounded_state)
        crashed = not landed
        stopped_state = replace(grounded_state, vx=0.0, vz=0.0, omega=0.0)
        return StepResult(
            state=stopped_state,
            terminated=True,
            landed=landed,
            crashed=crashed,
            impact_state=grounded_state,
        )

    def _is_soft_landing(self, state: State) -> bool:
        """Check whether the grounded state satisfies all landing thresholds."""

        return (
            abs(state.x) <= self._params.max_landing_x
            and abs(state.vz) <= self._params.max_landing_vz
            and abs(state.vx) <= self._params.max_landing_vx
            and abs(state.theta) <= self._params.max_landing_theta
            and abs(state.omega) <= self._params.max_landing_omega
        )
