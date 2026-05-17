"""Keyboard-driven controller implementation for pygame sessions."""

from __future__ import annotations

import pygame

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.infrastructure.config.models import ManualControlConfig


class PygameKeyboardManualController:
    """Manual controller driven by keyboard state."""

    def __init__(
        self,
        params: RocketParams,
        controls: ManualControlConfig,
    ) -> None:
        self._params = params
        self._controls = controls
        self._throttle = 0.0
        self._engine_gimbal = 0.0
        self._aero_steer = 0.0

    def reset(self) -> None:
        """Reset manual inputs to a neutral command state."""

        self._throttle = 0.0
        self._engine_gimbal = 0.0
        self._aero_steer = 0.0

    def sync_from_state(self, state: State) -> None:
        """Ignore simulation state; manual control only reacts to keyboard input."""

    def handle_pressed_keys(
        self,
        pressed_keys: pygame.key.ScancodeWrapper,
        dt: float,
    ) -> None:
        """Update manual controls from the currently pressed keys."""

        throttle_delta = 0.0
        if pressed_keys[pygame.K_UP]:
            throttle_delta += self._controls.throttle_rate * dt
        if pressed_keys[pygame.K_DOWN]:
            throttle_delta -= self._controls.throttle_rate * dt
        self._throttle = self._clamp(self._throttle + throttle_delta, 0.0, 1.0)

        if pressed_keys[pygame.K_LEFT]:
            self._engine_gimbal -= self._controls.engine_gimbal_rate * dt
            self._aero_steer -= self._controls.aero_steer_rate * dt
        elif pressed_keys[pygame.K_RIGHT]:
            self._engine_gimbal += self._controls.engine_gimbal_rate * dt
            self._aero_steer += self._controls.aero_steer_rate * dt
        else:
            self._engine_gimbal = self._move_towards(
                self._engine_gimbal,
                0.0,
                self._controls.steering_return_rate * dt,
            )
            self._aero_steer = self._move_towards(
                self._aero_steer,
                0.0,
                self._controls.steering_return_rate * 2.0 * dt,
            )

        self._engine_gimbal = self._clamp(
            self._engine_gimbal,
            -self._params.max_gimbal,
            self._params.max_gimbal,
        )
        self._aero_steer = self._clamp(self._aero_steer, -1.0, 1.0)

    def current_action(self) -> Action:
        """Return the latest action commanded by the user."""

        return Action(
            throttle=self._throttle,
            engine_gimbal=self._engine_gimbal,
            aero_steer=self._aero_steer,
        )

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        return min(max(value, low), high)

    @staticmethod
    def _move_towards(current: float, target: float, max_delta: float) -> float:
        if abs(target - current) <= max_delta:
            return target
        if current < target:
            return current + max_delta
        return current - max_delta
