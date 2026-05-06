from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Viewport:
    width: int = 1280
    height: int = 720
    padding: int = 48
    pixels_per_meter: float = 4.0
    hud_height: int = 84
    background: tuple[int, int, int] = (11, 15, 20)
    ground: tuple[int, int, int] = (228, 196, 127)
    grid: tuple[int, int, int] = (34, 43, 54)
    trajectory: tuple[int, int, int] = (77, 166, 255)
    booster: tuple[int, int, int] = (255, 112, 67)
    flame: tuple[int, int, int] = (255, 193, 7)
    text: tuple[int, int, int] = (236, 239, 244)
