"""Font palette used by the pygame HUD."""

from __future__ import annotations

from dataclasses import dataclass

import pygame


@dataclass(frozen=True, slots=True)
class HudFonts:
    """Centralized font palette for the pygame heads-up display."""

    title: pygame.font.Font
    metric: pygame.font.Font
    small: pygame.font.Font
    tiny: pygame.font.Font

    @classmethod
    def create(cls) -> HudFonts:
        """Build the font palette used by the HUD."""

        return cls(
            title=pygame.font.SysFont("consolas", 28, bold=True),
            metric=pygame.font.SysFont("consolas", 26, bold=True),
            small=pygame.font.SysFont("consolas", 18, bold=True),
            tiny=pygame.font.SysFont("consolas", 14, bold=True),
        )
