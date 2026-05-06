"""Heads-up display widgets used by the realtime pygame applications."""

from __future__ import annotations

import math

import pygame

from rocket_landing.infrastructure.rendering.pygame.telemetry import HudSnapshot
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


class HeadsUpDisplay:
    """Draw telemetry panels, gauges and control hints over the scene."""

    def __init__(self, viewport: Viewport) -> None:
        self._viewport = viewport

    def draw(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        snapshot: HudSnapshot,
    ) -> None:
        """Render the full HUD for the current frame."""

        self._draw_title(surface, fonts, snapshot)
        self._draw_left_metric_stack(surface, fonts, snapshot)
        self._draw_right_telemetry_panel(surface, fonts, snapshot)
        self._draw_bottom_status_bar(surface, fonts, snapshot)
        self._draw_throttle_gauge(surface, fonts, snapshot)
        self._draw_gimbal_gauge(surface, fonts, snapshot)

    def _draw_title(self, surface: pygame.Surface, fonts: HudFonts, snapshot: HudSnapshot) -> None:
        title = fonts.title.render(snapshot.title, True, self._viewport.text)
        subtitle = fonts.small.render(
            (
                f"{snapshot.mode.upper()}  |  "
                f"{snapshot.controller_name.upper()}  |  "
                f"{snapshot.status_text}"
            ),
            True,
            self._status_color(snapshot),
        )
        surface.blit(title, (28, 18))
        surface.blit(subtitle, (30, 52))

    def _draw_left_metric_stack(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        snapshot: HudSnapshot,
    ) -> None:
        card_width = 210
        card_height = 84
        start_x = 22
        start_y = 92
        gap = 12
        metrics = [
            ("ALTITUDE", f"{snapshot.state.z:07.1f} m", self._viewport.accent),
            ("VITESSE", f"{snapshot.state.speed:06.1f} m/s", self._viewport.accent),
            (
                "VITESSE VERT",
                f"{snapshot.state.vz:+06.1f} m/s",
                self._velocity_color(snapshot.state.vz),
            ),
            (
                "CARBURANT",
                f"{snapshot.fuel_ratio * 100:05.1f}%",
                self._fuel_color(snapshot.fuel_ratio),
            ),
        ]
        for index, (label, value, color) in enumerate(metrics):
            rect = pygame.Rect(
                start_x,
                start_y + index * (card_height + gap),
                card_width,
                card_height,
            )
            self._draw_panel(surface, rect)
            label_surface = fonts.small.render(label, True, self._viewport.text)
            value_surface = fonts.metric.render(value, True, color)
            surface.blit(label_surface, (rect.x + 14, rect.y + 12))
            surface.blit(value_surface, (rect.x + 14, rect.y + 38))

    def _draw_right_telemetry_panel(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        snapshot: HudSnapshot,
    ) -> None:
        panel = pygame.Rect(self._viewport.width - 312, 26, 286, 246)
        self._draw_panel(surface, panel)

        header = fonts.small.render("TELEMETRIE", True, self._viewport.text)
        surface.blit(header, (panel.x + 18, panel.y + 12))

        lines = [
            ("CONTROLE", snapshot.controller_name.upper()),
            ("STATUT", snapshot.status_text),
            ("PAS", snapshot.steps_label),
            ("PITCH", f"{math.degrees(snapshot.state.theta):+05.1f} deg"),
            ("GIMBAL", f"{math.degrees(snapshot.action.gimbal):+05.1f} deg"),
            ("POUSSEE", f"{snapshot.action.throttle * 100:05.1f}%"),
            ("TEMPS", self._format_time(snapshot.elapsed_time)),
            ("PAUSE", "OUI" if snapshot.paused else "NON"),
        ]
        y = panel.y + 52
        for label, value in lines:
            label_surface = fonts.small.render(label, True, self._viewport.text)
            value_surface = fonts.small.render(value, True, self._status_color(snapshot))
            surface.blit(label_surface, (panel.x + 18, y))
            surface.blit(value_surface, (panel.right - value_surface.get_width() - 18, y))
            y += 24

        extra_y = y + 8
        for line in snapshot.extra_lines[:2]:
            extra_surface = fonts.tiny.render(line, True, self._viewport.text)
            surface.blit(extra_surface, (panel.x + 18, extra_y))
            extra_y += 18

    def _draw_bottom_status_bar(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        snapshot: HudSnapshot,
    ) -> None:
        bar = pygame.Rect(20, self._viewport.height - 88, self._viewport.width - 40, 68)
        self._draw_panel(surface, bar)
        hint = fonts.small.render(
            (
                "TAB switch controller   SPACE pause   R reset   ESC quit   "
                "SHIFT precision   X cut throttle   C center gimbal"
            ),
            True,
            self._viewport.text,
        )
        surface.blit(hint, (bar.x + 18, bar.y + 12))
        line2 = fonts.small.render(
            (
                "Manual: W/S or Up/Down throttle, A/D or Left/Right gimbal. "
                "Baseline can be used as a landing reference."
            ),
            True,
            self._viewport.text,
        )
        surface.blit(line2, (bar.x + 18, bar.y + 38))

    def _draw_throttle_gauge(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        snapshot: HudSnapshot,
    ) -> None:
        gauge = pygame.Rect(self._viewport.width - 104, self._viewport.height - 284, 56, 160)
        self._draw_panel(surface, gauge)
        title = fonts.tiny.render("POUSSEE", True, self._viewport.text)
        surface.blit(title, (gauge.x + 6, gauge.y + 8))

        inner = gauge.inflate(-20, -36)
        inner.top += 18
        pygame.draw.rect(surface, (18, 25, 18), inner, border_radius=4)
        fill_height = int(round(inner.height * snapshot.action.throttle))
        fill_rect = pygame.Rect(
            inner.x + 4,
            inner.bottom - fill_height,
            inner.width - 8,
            fill_height,
        )
        pygame.draw.rect(
            surface,
            self._fuel_color(snapshot.action.throttle),
            fill_rect,
            border_radius=4,
        )
        percent = fonts.small.render(
            f"{snapshot.action.throttle * 100:03.0f}%",
            True,
            self._viewport.text,
        )
        surface.blit(percent, (gauge.centerx - (percent.get_width() // 2), gauge.bottom - 28))

    def _draw_gimbal_gauge(
        self,
        surface: pygame.Surface,
        fonts: HudFonts,
        snapshot: HudSnapshot,
    ) -> None:
        gauge = pygame.Rect(self._viewport.width - 326, self._viewport.height - 112, 230, 48)
        self._draw_panel(surface, gauge)
        label = fonts.tiny.render("GIMBAL", True, self._viewport.text)
        surface.blit(label, (gauge.x + 10, gauge.y + 8))

        track = pygame.Rect(gauge.x + 76, gauge.y + 18, 136, 12)
        pygame.draw.rect(surface, (24, 30, 40), track, border_radius=6)
        pygame.draw.line(surface, self._viewport.text, track.midleft, track.midright, 2)
        center_x = track.centerx
        ratio = snapshot.action.gimbal / max(0.001, math.radians(8.6))
        ratio = max(-1.0, min(1.0, ratio))
        marker_x = int(round(center_x + (ratio * (track.width / 2 - 8))))
        pygame.draw.circle(surface, self._viewport.accent_warm, (marker_x, track.centery), 8)
        value = fonts.small.render(
            f"{math.degrees(snapshot.action.gimbal):+05.1f} deg",
            True,
            self._viewport.text,
        )
        surface.blit(value, (gauge.x + 76, gauge.y + 28))

    def _draw_panel(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        shadow_rect = rect.move(4, 4)
        shadow = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        shadow.fill((*self._viewport.panel_shadow, 96))
        surface.blit(shadow, shadow_rect)

        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((*self._viewport.panel_bg, 230))
        pygame.draw.rect(panel, self._viewport.panel_border, panel.get_rect(), 3)
        surface.blit(panel, rect)

    def _status_color(self, snapshot: HudSnapshot) -> tuple[int, int, int]:
        if snapshot.status_text in {"CRASH", "ABORT"}:
            return self._viewport.warning
        if snapshot.status_text in {"LANDED", "STABLE"}:
            return self._viewport.accent
        return self._viewport.accent_warm

    def _velocity_color(self, vertical_speed: float) -> tuple[int, int, int]:
        if vertical_speed < -12.0:
            return self._viewport.warning
        if vertical_speed < -4.0:
            return self._viewport.accent_warm
        return self._viewport.accent

    def _fuel_color(self, ratio: float) -> tuple[int, int, int]:
        if ratio < 0.20:
            return self._viewport.warning
        if ratio < 0.45:
            return self._viewport.accent_warm
        return self._viewport.accent

    def _format_time(self, elapsed: float) -> str:
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        centiseconds = int((elapsed - int(elapsed)) * 100)
        return f"{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


class HudFonts:
    """Centralized font palette for the pygame heads-up display."""

    def __init__(self) -> None:
        self.title = pygame.font.SysFont("consolas", 28, bold=True)
        self.metric = pygame.font.SysFont("consolas", 26, bold=True)
        self.small = pygame.font.SysFont("consolas", 18, bold=True)
        self.tiny = pygame.font.SysFont("consolas", 14, bold=True)
