"""Matplotlib-based plotting helpers for offline trajectory inspection."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.physics.geometry import BoosterGeometry
from rocket_landing.domain.simulation.history import SimulationHistory


class MatplotlibTrajectoryPlotter:
    """Static trajectory plotter for analysis and export."""

    def __init__(self, params: RocketParams) -> None:
        self._geometry = BoosterGeometry(params)

    def render(
        self,
        history: SimulationHistory,
        *,
        title: str = "Rocket landing trajectory",
        output_path: str | None = None,
    ) -> None:
        """Render or export a static view of a recorded trajectory."""

        if history.is_empty():
            raise ValueError("history must contain at least one state")

        xs = [state.x for state in history.states]
        zs = [state.z for state in history.states]
        bottom, top = self._geometry.segment_endpoints(history.states[-1])

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.axhline(0.0, color="black", linewidth=1.2, linestyle="--")
        axis.plot(xs, zs, color="tab:blue", linewidth=2.0, label="Trajectory")
        axis.plot(
            [bottom[0], top[0]],
            [bottom[1], top[1]],
            color="tab:red",
            linewidth=3.0,
            label="Final attitude",
        )
        axis.scatter([xs[0]], [zs[0]], color="tab:green", s=60, label="Start")
        axis.scatter([xs[-1]], [zs[-1]], color="tab:orange", s=60, label="End")
        axis.set_xlabel("Horizontal position x (m)")
        axis.set_ylabel("Altitude z (m)")
        axis.set_title(title)
        axis.legend()
        axis.grid(True, alpha=0.3)
        axis.set_aspect("equal", adjustable="box")
        axis.set_ylim(bottom=-5.0)

        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            return

        plt.show()
