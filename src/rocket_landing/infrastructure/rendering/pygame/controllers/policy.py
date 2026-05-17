"""Policy-backed controller for pygame simulation sessions."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pygame

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.rl.env import decode_action

PolicyFn = Callable[[np.ndarray], np.ndarray]
ObservationFn = Callable[[State], np.ndarray]


class PolicyController:
    """Adapt a learned policy to the common pygame controller contract."""

    def __init__(
        self,
        params: RocketParams,
        *,
        policy_fn: PolicyFn,
        observation_fn: ObservationFn,
    ) -> None:
        self._params = params
        self._policy_fn = policy_fn
        self._observation_fn = observation_fn
        self._observation: np.ndarray | None = None

    def reset(self) -> None:
        """Forget any cached observation until a new episode is provided."""

        self._observation = None

    def sync_from_state(self, state: State) -> None:
        """Provide the latest simulator state to the controller."""

        self._observation = np.asarray(self._observation_fn(state), dtype=np.float32)

    def handle_pressed_keys(
        self,
        pressed_keys: pygame.key.ScancodeWrapper,
        dt: float,
    ) -> None:
        """Ignore keyboard state; the policy is autonomous."""

    def current_action(self) -> Action:
        """Return the current policy action decoded to the domain action space."""

        if self._observation is None:
            raise RuntimeError("policy controller requires an observation before stepping")
        raw_action = np.asarray(self._policy_fn(self._observation), dtype=np.float32)
        return decode_action(raw_action, self._params)
