"""SB3 model factory helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def build_sac_model(
    env: Any,
    *,
    config: Any,
    tensorboard_log_dir: Path | None = None,
) -> Any:
    """Build a SAC model bound to the provided vectorized environment."""

    from stable_baselines3 import SAC

    tensorboard_log = None
    if tensorboard_log_dir is not None and importlib.util.find_spec("tensorboard") is not None:
        tensorboard_log = tensorboard_log_dir.as_posix()

    return SAC(
        policy=config.sac.policy,
        env=env,
        learning_rate=config.sac.learning_rate,
        buffer_size=config.sac.buffer_size,
        learning_starts=config.sac.learning_starts,
        batch_size=config.sac.batch_size,
        tau=config.sac.tau,
        gamma=config.sac.gamma,
        train_freq=config.sac.train_freq,
        gradient_steps=config.sac.gradient_steps,
        ent_coef=config.sac.ent_coef,
        verbose=config.verbose,
        device=config.sac.device,
        seed=config.seed,
        tensorboard_log=tensorboard_log,
    )
