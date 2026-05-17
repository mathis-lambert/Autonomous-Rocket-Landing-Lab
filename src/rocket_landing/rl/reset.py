"""Initial-state sampling for RL episodes."""

from __future__ import annotations

import numpy as np

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.rl.types import InitialStateDistribution


def build_nominal_initial_state(params: RocketParams) -> State:
    """Return a deterministic spawn state centered over the target."""

    return State(
        x=params.target_x,
        z=120.0,
        vx=0.0,
        vz=-15.0,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )


def sample_initial_state(
    params: RocketParams,
    distribution: InitialStateDistribution,
    rng: np.random.Generator,
) -> State:
    """Sample one valid state from the configured distribution."""

    fuel_ratio = rng.uniform(distribution.fuel_ratio.low, distribution.fuel_ratio.high)
    return State(
        x=params.target_x + rng.uniform(distribution.dx.low, distribution.dx.high),
        z=rng.uniform(distribution.z.low, distribution.z.high),
        vx=rng.uniform(distribution.vx.low, distribution.vx.high),
        vz=rng.uniform(distribution.vz.low, distribution.vz.high),
        theta=rng.uniform(distribution.theta.low, distribution.theta.high),
        omega=rng.uniform(distribution.omega.low, distribution.omega.high),
        fuel=max(0.0, min(params.initial_fuel, params.initial_fuel * fuel_ratio)),
    )

