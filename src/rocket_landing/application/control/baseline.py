"""Deterministic baseline controller used before introducing reinforcement learning.

The baseline is intentionally simple and hand-tuned.  Its role is not to be
optimal, but to provide:
- a sanity check that the vehicle is controllable
- a comparison point for later RL agents
- a readable reference policy for debugging the physics model
"""

from __future__ import annotations

from rocket_landing.application.control.controller import FlightController
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State

HIGH_ALTITUDE_THRESHOLD = 85.0
MID_ALTITUDE_THRESHOLD = 45.0
LOW_ALTITUDE_THRESHOLD = 20.0

HIGH_ALTITUDE_TARGET_VZ = -20.0
MID_ALTITUDE_TARGET_VZ = -12.0
LOW_ALTITUDE_TARGET_VZ = -6.0
FINAL_DESCENT_MIN_SPEED = 1.0
FINAL_DESCENT_GAIN = 0.14

HORIZONTAL_POSITION_GAIN = 0.0085
HORIZONTAL_VELOCITY_GAIN = 0.060
MAX_TARGET_THETA = 0.22

THETA_PROPORTIONAL_GAIN = 5.8
THETA_DAMPING_GAIN = 1.75
VERTICAL_SPEED_GAIN = 0.032
ATTITUDE_FEEDFORWARD_GAIN = 0.030


class BaselineLandingController(FlightController):
    """A simple deterministic landing controller used as a baseline.

    The controller combines:
    - a lateral guidance law turning horizontal error into a target attitude
    - an attitude stabilizer turning that target into a gimbal command
    - a piecewise vertical-speed schedule converted into throttle
    """

    def __init__(self, params: RocketParams) -> None:
        self._params = params

    @property
    def name(self) -> str:
        """Return the identifier displayed by the CLI and HUD."""

        return "baseline"

    def reset(self, state: State) -> None:
        """Reset cached targets when a new episode starts."""

        del state

    def compute_action(self, state: State, dt: float) -> Action:
        """Compute a stabilizing throttle and gimbal command.

        Args:
            state: Current simulated booster state.
            dt: Simulation step size, unused for this simple controller.
        """

        del dt
        target_theta = self._desired_theta(state)
        target_vz = self._desired_vertical_speed(state)

        theta_error = target_theta - state.theta
        gimbal = (THETA_PROPORTIONAL_GAIN * theta_error) - (THETA_DAMPING_GAIN * state.omega)
        gimbal = self._clamp(gimbal, -self._params.max_gimbal, self._params.max_gimbal)

        hover_throttle = self._hover_throttle(state)
        vertical_error = target_vz - state.vz
        throttle = (
            hover_throttle
            + (VERTICAL_SPEED_GAIN * vertical_error)
            + (ATTITUDE_FEEDFORWARD_GAIN * abs(target_theta))
        )
        throttle = self._clamp(throttle, 0.0, 1.0)
        return Action(throttle=throttle, gimbal=gimbal)

    def _desired_theta(self, state: State) -> float:
        """Turn horizontal position and drift into an attitude target."""

        target_theta = -(HORIZONTAL_POSITION_GAIN * state.x) - (HORIZONTAL_VELOCITY_GAIN * state.vx)
        return self._clamp(target_theta, -MAX_TARGET_THETA, MAX_TARGET_THETA)

    def _desired_vertical_speed(self, state: State) -> float:
        """Use a piecewise descent profile that slows near the landing pad."""

        if state.z > HIGH_ALTITUDE_THRESHOLD:
            return HIGH_ALTITUDE_TARGET_VZ
        if state.z > MID_ALTITUDE_THRESHOLD:
            return MID_ALTITUDE_TARGET_VZ
        if state.z > LOW_ALTITUDE_THRESHOLD:
            return LOW_ALTITUDE_TARGET_VZ
        return -max(FINAL_DESCENT_MIN_SPEED, FINAL_DESCENT_GAIN * state.z)

    def _hover_throttle(self, state: State) -> float:
        """Return the throttle required to exactly counteract gravity."""

        mass = self._params.dry_mass + state.fuel
        return (mass * self._params.gravity) / self._params.max_thrust

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        """Clamp a scalar between the provided lower and upper bounds."""

        return min(max(value, low), high)
