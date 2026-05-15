"""Typed episode summaries shared by evaluation and environment info."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """Compact episode-level telemetry for logging and evaluation."""

    steps: int
    elapsed_time: float
    return_total: float
    landed: bool
    crashed: bool
    truncated: bool
    final_dx: float
    final_altitude: float
    final_speed: float
    fuel_ratio: float

    def as_dict(self) -> dict[str, float | bool | int]:
        """Return a plain mapping suitable for Gymnasium ``info`` payloads."""

        return asdict(self)
