"""Rocket landing simulation package.

The package exposes the CLI entrypoint while keeping the internal architecture
split into ``domain``, ``application`` and ``infrastructure`` layers.
"""

from rocket_landing.infrastructure.cli.main import main

__all__ = ["main"]
