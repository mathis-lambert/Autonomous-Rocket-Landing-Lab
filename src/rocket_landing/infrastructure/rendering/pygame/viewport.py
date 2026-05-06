"""Viewport configuration and color palette for pygame renderers."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class Viewport:
    """Immutable viewport dimensions and styling tokens."""

    width: int = 1280
    height: int = 720
    padding: int = 24
    hud_height: int = 212
    min_pixels_per_meter: float = 1.15
    max_pixels_per_meter: float = 4.75
    background: tuple[int, int, int] = (8, 19, 36)
    text: tuple[int, int, int] = (235, 241, 255)
    accent: tuple[int, int, int] = (59, 224, 96)
    accent_warm: tuple[int, int, int] = (255, 188, 46)
    warning: tuple[int, int, int] = (255, 109, 82)
    panel_bg: tuple[int, int, int] = (8, 9, 12)
    panel_border: tuple[int, int, int] = (214, 220, 232)
    panel_shadow: tuple[int, int, int] = (0, 0, 0)
    trajectory: tuple[int, int, int] = (255, 255, 255)
    target_ring: tuple[int, int, int] = (255, 255, 255)
    velocity_vector: tuple[int, int, int] = (255, 210, 68)
    guide: tuple[int, int, int] = (255, 255, 255)

    def resized(self, width: int, height: int) -> Viewport:
        """Return a copy resized with the project's minimum window constraints."""

        return replace(
            self,
            width=max(960, width),
            height=max(540, height),
        )
