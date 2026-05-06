"""Small immutable telemetry snapshots consumed by the pygame HUD."""

from __future__ import annotations

from dataclasses import dataclass, field

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


@dataclass(frozen=True, slots=True)
class HudSnapshot:
    """Renderable telemetry state derived from a simulation frame."""

    title: str
    mode: str
    controller_name: str
    state: State
    action: Action
    elapsed_time: float
    fuel_ratio: float
    status_text: str
    steps_label: str
    paused: bool
    extra_lines: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_session(
        cls,
        *,
        params: RocketParams,
        mode: str,
        controller_name: str,
        state: State,
        action: Action,
        elapsed_time: float,
        status_text: str,
        steps_label: str,
        paused: bool,
        extra_lines: list[str] | tuple[str, ...] = (),
    ) -> HudSnapshot:
        """Build a HUD snapshot from the current session state."""

        fuel_ratio = 0.0
        if params.initial_fuel > 0.0:
            fuel_ratio = max(0.0, min(1.0, state.fuel / params.initial_fuel))
        return cls(
            title="ROCKET LANDING",
            mode=mode,
            controller_name=controller_name,
            state=state,
            action=action,
            elapsed_time=elapsed_time,
            fuel_ratio=fuel_ratio,
            status_text=status_text,
            steps_label=steps_label,
            paused=paused,
            extra_lines=tuple(extra_lines),
        )
