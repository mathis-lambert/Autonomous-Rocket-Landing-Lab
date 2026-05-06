"""Keyboard-driven controller implementation for the live pygame client."""

from __future__ import annotations

import pygame

from rocket_landing.application.control.controller import FlightController
from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State


class PygameKeyboardManualController(FlightController):
    """Manual controller driven by keyboard state."""

    def __init__(
        self,
        params: RocketParams,
        *,
        throttle_rate: float = 1.15,
        gimbal_rate: float = 1.25,
        gimbal_return_rate: float = 1.65,
    ) -> None:
        self._params = params
        self._throttle_rate = throttle_rate
        self._gimbal_rate = gimbal_rate
        self._gimbal_return_rate = gimbal_return_rate
        self._throttle = 0.0
        self._gimbal = 0.0
        self._precision_mode = False

    @property
    def name(self) -> str:
        """Return the controller identifier shown in the UI."""

        return "manual"

    def reset(self, state: State) -> None:
        """Reset manual inputs to a neutral command state."""

        del state
        self._throttle = 0.0
        self._gimbal = 0.0
        self._precision_mode = False

    def update_from_pressed_keys(
        self,
        pressed_keys: pygame.key.ScancodeWrapper,
        dt: float,
    ) -> None:
        """Update manual controls from the currently pressed keys."""

        precision_multiplier = 0.35 if self._precision_mode else 1.0
        throttle_delta = 0.0
        if pressed_keys[pygame.K_UP] or pressed_keys[pygame.K_w]:
            throttle_delta += self._throttle_rate * precision_multiplier * dt
        if pressed_keys[pygame.K_DOWN] or pressed_keys[pygame.K_s]:
            throttle_delta -= self._throttle_rate * precision_multiplier * dt
        self._throttle = self._clamp(self._throttle + throttle_delta, 0.0, 1.0)

        if pressed_keys[pygame.K_LEFT] or pressed_keys[pygame.K_a]:
            self._gimbal -= self._gimbal_rate * precision_multiplier * dt
        elif pressed_keys[pygame.K_RIGHT] or pressed_keys[pygame.K_d]:
            self._gimbal += self._gimbal_rate * precision_multiplier * dt
        else:
            self._gimbal = self._move_towards(self._gimbal, 0.0, self._gimbal_return_rate * dt)

        self._gimbal = self._clamp(self._gimbal, -self._params.max_gimbal, self._params.max_gimbal)

    def handle_event(self, event: pygame.event.Event) -> None:
        """Handle discrete key events such as trims and precision mode."""

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                self._throttle = 0.0
            elif event.key == pygame.K_c:
                self._gimbal = 0.0
            elif event.key == pygame.K_z:
                self._throttle = self._clamp(self._throttle + 0.15, 0.0, 1.0)
            elif event.key == pygame.K_f:
                self._throttle = self._clamp(self._throttle - 0.15, 0.0, 1.0)
            elif event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
                self._precision_mode = True
        elif event.type == pygame.KEYUP and event.key in (pygame.K_LSHIFT, pygame.K_RSHIFT):
            self._precision_mode = False

    def compute_action(self, state: State, dt: float) -> Action:
        """Return the latest manual command without inspecting the state."""

        del state, dt
        return Action(throttle=self._throttle, gimbal=self._gimbal)

    def status_lines(self) -> list[str]:
        """Return short manual-control hints for the HUD footer."""

        return [
            (
                f"controller = manual   throttle = {self._throttle:4.2f}   "
                f"gimbal = {self._gimbal:6.3f} rad   precision = {self._precision_mode}"
            ),
            (
                "controls: [W/Up][S/Down] throttle  [A/Left][D/Right] gimbal  "
                "[Shift] precision  [Z] +15%  [F] -15%  [X] cut  [C] center"
            ),
        ]

    @staticmethod
    def _clamp(value: float, low: float, high: float) -> float:
        """Clamp a scalar between the provided bounds."""

        return min(max(value, low), high)

    @staticmethod
    def _move_towards(current: float, target: float, max_delta: float) -> float:
        """Move a scalar toward a target by at most ``max_delta``."""

        if abs(target - current) <= max_delta:
            return target
        if current < target:
            return current + max_delta
        return current - max_delta
