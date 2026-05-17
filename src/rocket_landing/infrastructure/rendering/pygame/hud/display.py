"""Compact heads-up display used by the pygame applications."""

from __future__ import annotations

import math

import pygame

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.infrastructure.rendering.pygame.hud.fonts import HudFonts
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

HUD_MARGIN = 18
TOP_BAR_HEIGHT = 58
STATUS_PANEL_WIDTH = 300
HELP_BAR_HEIGHT = 30
PANEL_ALPHA = 178
PANEL_BORDER_ALPHA = 120
PANEL_RADIUS = 4
CHIP_GAP = 18
TITLE_WIDTH = 170
LABEL_VALUE_GAP = 6

CRITICAL_VERTICAL_SPEED = -12.0
WARNING_VERTICAL_SPEED = -4.0
LOW_FUEL_RATIO = 0.20
WARNING_FUEL_RATIO = 0.45

HELP_TEXT = "ARROWS fly   F3 forces   SPACE pause   R reset   ESC quit"


class HeadsUpDisplay:
    """Draw compact flight instrumentation over the scene."""

    def __init__(self, params: RocketParams, viewport: Viewport) -> None:
        self._params = params
        self._viewport = viewport

    def draw(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        *,
        state: State,
        action: Action,
        elapsed_time: float,
        status_text: str,
        steps_label: str,
        paused: bool,
        debug_forces: bool,
    ) -> None:
        """Render the HUD for the current frame."""

        fuel_ratio = 0.0
        if self._params.initial_fuel > 0.0:
            fuel_ratio = max(0.0, min(1.0, state.fuel / self._params.initial_fuel))

        self._draw_top_bar(
            surface,
            fonts,
            state=state,
            action=action,
            fuel_ratio=fuel_ratio,
            status_text=status_text,
        )
        self._draw_status_panel(
            surface,
            fonts,
            elapsed_time=elapsed_time,
            steps_label=steps_label,
            paused=paused,
            debug_forces=debug_forces,
        )
        self._draw_help_bar(surface, fonts)

    def _draw_top_bar(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        *,
        state: State,
        action: Action,
        fuel_ratio: float,
        status_text: str,
    ) -> None:
        available_width = self._viewport.width - STATUS_PANEL_WIDTH - (HUD_MARGIN * 3)
        bar = pygame.Rect(HUD_MARGIN, HUD_MARGIN, available_width, TOP_BAR_HEIGHT)
        self._draw_panel(surface, bar)

        title = fonts.title.render("ROCKET", True, self._viewport.text)
        status = fonts.tiny.render(status_text, True, self._status_color(status_text))
        surface.blit(title, (bar.x + 14, bar.y + 10))
        surface.blit(status, (bar.x + 15, bar.y + 35))

        ground_clearance = state.ground_clearance(self._params.length)
        metrics = [
            ("AGL", f"{ground_clearance:,.0f} m", self._viewport.accent),
            ("V", f"{state.speed:,.1f} m/s", self._viewport.accent),
            ("VZ", f"{state.vz:+.1f}", self._velocity_color(state.vz)),
            ("FUEL", f"{fuel_ratio * 100:.0f}%", self._fuel_color(fuel_ratio)),
            ("THR", f"{action.throttle * 100:.0f}%", self._fuel_color(action.throttle)),
            (
                "ENG",
                f"{math.degrees(action.engine_gimbal):+.1f}deg",
                self._viewport.accent_warm,
            ),
            ("AERO", f"{action.aero_steer * 100:+.0f}%", self._viewport.accent_warm),
        ]

        x = bar.x + TITLE_WIDTH
        y = bar.y + 18
        for label, value, color in metrics:
            x = self._draw_metric(surface, fonts, x, y, label, value, color)
            x += CHIP_GAP

    def _draw_status_panel(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        *,
        elapsed_time: float,
        steps_label: str,
        paused: bool,
        debug_forces: bool,
    ) -> None:
        panel = pygame.Rect(
            self._viewport.width - STATUS_PANEL_WIDTH - HUD_MARGIN,
            HUD_MARGIN,
            STATUS_PANEL_WIDTH,
            TOP_BAR_HEIGHT,
        )
        self._draw_panel(surface, panel)

        left_lines = [("SIM", "LIVE"), ("INPUT", "ARROWS")]
        right_lines = [
            ("TIME", self._format_time(elapsed_time)),
            ("STEP", steps_label),
        ]
        self._draw_tiny_lines(surface, fonts, panel.x + 14, panel.y + 10, left_lines)
        self._draw_tiny_lines(surface, fonts, panel.x + 158, panel.y + 10, right_lines)

        if paused:
            paused_surface = fonts.tiny.render("PAUSED", True, self._viewport.accent_warm)
            surface.blit(
                paused_surface,
                (panel.right - paused_surface.get_width() - 14, panel.y + 36),
            )
        elif debug_forces:
            debug_surface = fonts.tiny.render("FORCES", True, self._viewport.accent_warm)
            surface.blit(
                debug_surface,
                (panel.right - debug_surface.get_width() - 14, panel.y + 36),
            )

    def _draw_help_bar(self, surface: pygame.Surface, fonts: HudFonts) -> None:
        text = fonts.tiny.render(HELP_TEXT, True, self._viewport.text)
        bar_width = min(self._viewport.width - (HUD_MARGIN * 2), text.get_width() + 28)
        bar = pygame.Rect(
            HUD_MARGIN,
            self._viewport.height - HELP_BAR_HEIGHT - HUD_MARGIN,
            bar_width,
            HELP_BAR_HEIGHT,
        )
        self._draw_panel(surface, bar)
        surface.blit(text, (bar.x + 14, bar.y + 8))

    def _draw_metric(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        x: int,
        y: int,
        label: str,
        value: str,
        color: tuple[int, int, int],
    ) -> int:
        label_surface = fonts.tiny.render(label, True, self._viewport.text)
        value_surface = fonts.metric.render(value, True, color)
        surface.blit(label_surface, (x, y + 4))
        value_x = x + label_surface.get_width() + LABEL_VALUE_GAP
        surface.blit(value_surface, (value_x, y))
        return value_x + value_surface.get_width()

    def _draw_tiny_lines(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        x: int,
        y: int,
        lines: list[tuple[str, str]],
    ) -> None:
        for index, (label, value) in enumerate(lines):
            line_y = y + (index * 20)
            label_surface = fonts.tiny.render(label, True, self._viewport.text)
            value_surface = fonts.tiny.render(value, True, self._viewport.accent)
            surface.blit(label_surface, (x, line_y))
            surface.blit(value_surface, (x + 48, line_y))

    def _draw_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((*self._viewport.panel_bg, PANEL_ALPHA))
        pygame.draw.rect(
            panel,
            (*self._viewport.panel_border, PANEL_BORDER_ALPHA),
            panel.get_rect(),
            1,
            border_radius=PANEL_RADIUS,
        )
        surface.blit(panel, rect)

    def _status_color(self, status_text: str) -> tuple[int, int, int]:
        if status_text in {"CRASH", "ABORT"}:
            return self._viewport.warning
        if status_text in {"LANDED", "STABLE"}:
            return self._viewport.accent
        return self._viewport.accent_warm

    def _velocity_color(self, vertical_speed: float) -> tuple[int, int, int]:
        if vertical_speed < CRITICAL_VERTICAL_SPEED:
            return self._viewport.warning
        if vertical_speed < WARNING_VERTICAL_SPEED:
            return self._viewport.accent_warm
        return self._viewport.accent

    def _fuel_color(self, ratio: float) -> tuple[int, int, int]:
        if ratio < LOW_FUEL_RATIO:
            return self._viewport.warning
        if ratio < WARNING_FUEL_RATIO:
            return self._viewport.accent_warm
        return self._viewport.accent

    def _format_time(self, elapsed: float) -> str:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        centiseconds = int((elapsed - int(elapsed)) * 100)
        return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
