"""Heads-up display widgets used by the realtime pygame applications."""

from __future__ import annotations

import math

import pygame

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.infrastructure.rendering.pygame.hud.fonts import HudFonts
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

TITLE_POSITION = (28, 18)
SUBTITLE_POSITION = (30, 52)

METRIC_CARD_WIDTH = 210
METRIC_CARD_HEIGHT = 84
METRIC_CARD_START_X = 22
METRIC_CARD_START_Y = 92
METRIC_CARD_GAP = 12
METRIC_LABEL_OFFSET = (14, 12)
METRIC_VALUE_OFFSET = (14, 38)

TELEMETRY_PANEL_MARGIN_TOP = 26
TELEMETRY_PANEL_MARGIN_RIGHT = 26
TELEMETRY_PANEL_WIDTH = 286
TELEMETRY_PANEL_HEIGHT = 246
TELEMETRY_HEADER_OFFSET = (18, 12)
TELEMETRY_LINES_START_Y = 52
TELEMETRY_LINE_HEIGHT = 24
TELEMETRY_EXTRA_GAP = 8
TELEMETRY_EXTRA_LINE_HEIGHT = 18
TELEMETRY_MAX_EXTRA_LINES = 2

STATUS_BAR_MARGIN_X = 20
STATUS_BAR_MARGIN_BOTTOM = 20
STATUS_BAR_HEIGHT = 68
STATUS_BAR_TEXT_X = 18
STATUS_BAR_HINT_Y = 12
STATUS_BAR_SECONDARY_Y = 38

THROTTLE_GAUGE_MARGIN_RIGHT = 48
THROTTLE_GAUGE_MARGIN_BOTTOM = 284
THROTTLE_GAUGE_WIDTH = 56
THROTTLE_GAUGE_HEIGHT = 160
THROTTLE_GAUGE_TITLE_OFFSET = (6, 8)
THROTTLE_GAUGE_INNER_INSET_X = 20
THROTTLE_GAUGE_INNER_INSET_Y = 36
THROTTLE_GAUGE_INNER_TOP_OFFSET = 18
THROTTLE_GAUGE_FILL_INSET = 4
THROTTLE_GAUGE_PERCENT_BOTTOM = 28

GIMBAL_GAUGE_MARGIN_RIGHT = 326
GIMBAL_GAUGE_MARGIN_BOTTOM = 112
GIMBAL_GAUGE_WIDTH = 230
GIMBAL_GAUGE_HEIGHT = 48
GIMBAL_GAUGE_LABEL_OFFSET = (10, 8)
GIMBAL_TRACK_OFFSET = (76, 18)
GIMBAL_TRACK_SIZE = (136, 12)
GIMBAL_TRACK_MARKER_PADDING = 8
GIMBAL_VALUE_OFFSET = (76, 28)

PANEL_SHADOW_OFFSET = (4, 4)
PANEL_SHADOW_ALPHA = 96
PANEL_FILL_ALPHA = 230
PANEL_BORDER_WIDTH = 3

THROTTLE_GAUGE_BACKGROUND = (18, 25, 18)
GIMBAL_TRACK_BACKGROUND = (24, 30, 40)

CRITICAL_VERTICAL_SPEED = -12.0
WARNING_VERTICAL_SPEED = -4.0
LOW_FUEL_RATIO = 0.20
WARNING_FUEL_RATIO = 0.45

HINT_TEXT = (
    "TAB switch controller   SPACE pause   R reset   ESC quit   "
    "SHIFT precision   X cut throttle   C center gimbal"
)
SECONDARY_HINT_TEXT = "Manual: W/S or Up/Down throttle, A/D or Left/Right gimbal."


class HeadsUpDisplay:
    """Draw telemetry panels, gauges and control hints over the scene."""

    def __init__(self, params: RocketParams, viewport: Viewport) -> None:
        self._params = params
        self._viewport = viewport

    def draw(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        *,
        mode: str,
        controller_name: str,
        state: State,
        action: Action,
        elapsed_time: float,
        status_text: str,
        steps_label: str,
        paused: bool,
        extra_lines: list[str] | tuple[str, ...] = (),
    ) -> None:
        """Render the full HUD for the current frame."""

        fuel_ratio = 0.0
        if self._params.initial_fuel > 0.0:
            fuel_ratio = max(0.0, min(1.0, state.fuel / self._params.initial_fuel))

        title = fonts.title.render("ROCKET LANDING", True, self._viewport.text)
        subtitle = fonts.small.render(
            f"{mode.upper()}  |  {controller_name.upper()}  |  {status_text}",
            True,
            self._status_color(status_text),
        )
        surface.blit(title, TITLE_POSITION)
        surface.blit(subtitle, SUBTITLE_POSITION)

        metrics = [
            ("ALTITUDE", f"{state.z:07.1f} m", self._viewport.accent),
            ("VITESSE", f"{state.speed:06.1f} m/s", self._viewport.accent),
            (
                "VITESSE VERT",
                f"{state.vz:+06.1f} m/s",
                self._velocity_color(state.vz),
            ),
            (
                "CARBURANT",
                f"{fuel_ratio * 100:05.1f}%",
                self._fuel_color(fuel_ratio),
            ),
        ]
        for index, (label, value, color) in enumerate(metrics):
            rect = pygame.Rect(
                METRIC_CARD_START_X,
                METRIC_CARD_START_Y + index * (METRIC_CARD_HEIGHT + METRIC_CARD_GAP),
                METRIC_CARD_WIDTH,
                METRIC_CARD_HEIGHT,
            )
            self._draw_panel(surface, rect)
            label_surface = fonts.small.render(label, True, self._viewport.text)
            value_surface = fonts.metric.render(value, True, color)
            surface.blit(
                label_surface,
                (rect.x + METRIC_LABEL_OFFSET[0], rect.y + METRIC_LABEL_OFFSET[1]),
            )
            surface.blit(
                value_surface,
                (rect.x + METRIC_VALUE_OFFSET[0], rect.y + METRIC_VALUE_OFFSET[1]),
            )

        panel = pygame.Rect(
            self._viewport.width - TELEMETRY_PANEL_WIDTH - TELEMETRY_PANEL_MARGIN_RIGHT,
            TELEMETRY_PANEL_MARGIN_TOP,
            TELEMETRY_PANEL_WIDTH,
            TELEMETRY_PANEL_HEIGHT,
        )
        self._draw_panel(surface, panel)

        header = fonts.small.render("TELEMETRIE", True, self._viewport.text)
        surface.blit(
            header,
            (panel.x + TELEMETRY_HEADER_OFFSET[0], panel.y + TELEMETRY_HEADER_OFFSET[1]),
        )

        lines = [
            ("CONTROLE", controller_name.upper()),
            ("STATUT", status_text),
            ("PAS", steps_label),
            ("PITCH", f"{math.degrees(state.theta):+05.1f} deg"),
            ("GIMBAL", f"{math.degrees(action.gimbal):+05.1f} deg"),
            ("POUSSEE", f"{action.throttle * 100:05.1f}%"),
            ("TEMPS", self._format_time(elapsed_time)),
            ("PAUSE", "OUI" if paused else "NON"),
        ]
        y = panel.y + TELEMETRY_LINES_START_Y
        for label, value in lines:
            label_surface = fonts.small.render(label, True, self._viewport.text)
            value_surface = fonts.small.render(value, True, self._status_color(status_text))
            surface.blit(label_surface, (panel.x + TELEMETRY_HEADER_OFFSET[0], y))
            surface.blit(
                value_surface,
                (panel.right - value_surface.get_width() - TELEMETRY_HEADER_OFFSET[0], y),
            )
            y += TELEMETRY_LINE_HEIGHT

        extra_y = y + TELEMETRY_EXTRA_GAP
        for line in extra_lines[:TELEMETRY_MAX_EXTRA_LINES]:
            extra_surface = fonts.tiny.render(line, True, self._viewport.text)
            surface.blit(extra_surface, (panel.x + TELEMETRY_HEADER_OFFSET[0], extra_y))
            extra_y += TELEMETRY_EXTRA_LINE_HEIGHT

        bar = pygame.Rect(
            STATUS_BAR_MARGIN_X,
            self._viewport.height - STATUS_BAR_HEIGHT - STATUS_BAR_MARGIN_BOTTOM,
            self._viewport.width - (2 * STATUS_BAR_MARGIN_X),
            STATUS_BAR_HEIGHT,
        )
        self._draw_panel(surface, bar)
        hint = fonts.small.render(HINT_TEXT, True, self._viewport.text)
        surface.blit(hint, (bar.x + STATUS_BAR_TEXT_X, bar.y + STATUS_BAR_HINT_Y))
        line2 = fonts.small.render(
            SECONDARY_HINT_TEXT,
            True,
            self._viewport.text,
        )
        surface.blit(line2, (bar.x + STATUS_BAR_TEXT_X, bar.y + STATUS_BAR_SECONDARY_Y))

        gauge = pygame.Rect(
            self._viewport.width - THROTTLE_GAUGE_WIDTH - THROTTLE_GAUGE_MARGIN_RIGHT,
            self._viewport.height - THROTTLE_GAUGE_HEIGHT - THROTTLE_GAUGE_MARGIN_BOTTOM,
            THROTTLE_GAUGE_WIDTH,
            THROTTLE_GAUGE_HEIGHT,
        )
        self._draw_panel(surface, gauge)
        throttle_title = fonts.tiny.render("POUSSEE", True, self._viewport.text)
        surface.blit(
            throttle_title,
            (gauge.x + THROTTLE_GAUGE_TITLE_OFFSET[0], gauge.y + THROTTLE_GAUGE_TITLE_OFFSET[1]),
        )

        inner = gauge.inflate(-THROTTLE_GAUGE_INNER_INSET_X, -THROTTLE_GAUGE_INNER_INSET_Y)
        inner.top += THROTTLE_GAUGE_INNER_TOP_OFFSET
        pygame.draw.rect(surface, THROTTLE_GAUGE_BACKGROUND, inner, border_radius=4)
        fill_height = int(round(inner.height * action.throttle))
        fill_rect = pygame.Rect(
            inner.x + THROTTLE_GAUGE_FILL_INSET,
            inner.bottom - fill_height,
            inner.width - (2 * THROTTLE_GAUGE_FILL_INSET),
            fill_height,
        )
        pygame.draw.rect(
            surface,
            self._fuel_color(action.throttle),
            fill_rect,
            border_radius=4,
        )
        percent = fonts.small.render(
            f"{action.throttle * 100:03.0f}%",
            True,
            self._viewport.text,
        )
        surface.blit(
            percent,
            (
                gauge.centerx - (percent.get_width() // 2),
                gauge.bottom - THROTTLE_GAUGE_PERCENT_BOTTOM,
            ),
        )

        gauge = pygame.Rect(
            self._viewport.width - GIMBAL_GAUGE_WIDTH - GIMBAL_GAUGE_MARGIN_RIGHT,
            self._viewport.height - GIMBAL_GAUGE_HEIGHT - GIMBAL_GAUGE_MARGIN_BOTTOM,
            GIMBAL_GAUGE_WIDTH,
            GIMBAL_GAUGE_HEIGHT,
        )
        self._draw_panel(surface, gauge)
        label = fonts.tiny.render("GIMBAL", True, self._viewport.text)
        surface.blit(
            label,
            (gauge.x + GIMBAL_GAUGE_LABEL_OFFSET[0], gauge.y + GIMBAL_GAUGE_LABEL_OFFSET[1]),
        )

        track = pygame.Rect(
            gauge.x + GIMBAL_TRACK_OFFSET[0],
            gauge.y + GIMBAL_TRACK_OFFSET[1],
            GIMBAL_TRACK_SIZE[0],
            GIMBAL_TRACK_SIZE[1],
        )
        pygame.draw.rect(surface, GIMBAL_TRACK_BACKGROUND, track, border_radius=6)
        pygame.draw.line(surface, self._viewport.text, track.midleft, track.midright, 2)
        center_x = track.centerx
        ratio = action.gimbal / max(0.001, self._params.max_gimbal)
        ratio = max(-1.0, min(1.0, ratio))
        marker_x = int(round(center_x + (ratio * (track.width / 2 - GIMBAL_TRACK_MARKER_PADDING))))
        pygame.draw.circle(surface, self._viewport.accent_warm, (marker_x, track.centery), 8)
        value = fonts.small.render(
            f"{math.degrees(action.gimbal):+05.1f} deg",
            True,
            self._viewport.text,
        )
        surface.blit(value, (gauge.x + GIMBAL_VALUE_OFFSET[0], gauge.y + GIMBAL_VALUE_OFFSET[1]))

    def _draw_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        shadow_rect = rect.move(*PANEL_SHADOW_OFFSET)
        shadow = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        shadow.fill((*self._viewport.panel_shadow, PANEL_SHADOW_ALPHA))
        surface.blit(shadow, shadow_rect)

        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((*self._viewport.panel_bg, PANEL_FILL_ALPHA))
        pygame.draw.rect(panel, self._viewport.panel_border, panel.get_rect(), PANEL_BORDER_WIDTH)
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
