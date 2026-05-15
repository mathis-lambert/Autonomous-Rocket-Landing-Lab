"""Small SB3 callbacks used by the trainer."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_training_callbacks(stage_name_provider: Callable[[], str]) -> Any:
    """Build the callback stack used during training."""

    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    class StageLoggingCallback(BaseCallback):
        """Record the current curriculum stage in the SB3 logger."""

        def __init__(self) -> None:
            super().__init__(verbose=0)

        def _on_step(self) -> bool:
            self.logger.record("curriculum/stage_name", stage_name_provider())
            return True

    return CallbackList([StageLoggingCallback()])
