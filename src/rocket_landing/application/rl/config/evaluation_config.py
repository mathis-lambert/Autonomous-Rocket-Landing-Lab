"""Evaluation configuration shared by trainers and standalone scripts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Control how policies are evaluated against the environment."""

    episodes: int = 20
    deterministic: bool = True
    seed: int | None = None
