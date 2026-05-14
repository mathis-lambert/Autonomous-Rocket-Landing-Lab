"""Debug overlays for visualizing physics quantities in the pygame scene."""

from __future__ import annotations

import math

import pygame

from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.results import ForceBreakdown, ForceVector
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.aerodynamics import AerodynamicLoads
from rocket_landing.domain.physics.dynamics import BoosterDynamicsModel
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport

FORCE_MIN_MAGNITUDE_N = 1.0
FORCE_VECTOR_SCALE_PX_PER_KN = 0.065
FORCE_VECTOR_MIN_LENGTH_PX = 34.0
FORCE_VECTOR_MAX_LENGTH_PX = 260.0
FORCE_VECTOR_LINE_WIDTH = 4
FORCE_VECTOR_HEAD_LENGTH_PX = 16.0
FORCE_VECTOR_HEAD_HALF_WIDTH_PX = 8.0
FORCE_LABEL_MARGIN_X = 18
FORCE_LABEL_MARGIN_Y = 88
FORCE_LABEL_LINE_HEIGHT = 22
FORCE_PANEL_WIDTH = 360
FORCE_PANEL_PADDING = 12
FORCE_PANEL_RADIUS = 6
FORCE_PANEL_ALPHA = 176
FORCE_PANEL_BORDER_ALPHA = 110
ORIGIN_RADIUS_PX = 5
BODY_AXIS_LENGTH_SCALE = 0.7
BODY_AXIS_WIDTH = 2
BODY_AXIS_COLOR = (214, 220, 232)
AERODYNAMIC_VECTOR_COLOR = (106, 175, 255)
CONTROL_MARKER_COLOR = (255, 255, 255)
TEXT_SHADOW_COLOR = (0, 0, 0)
LABEL_OFFSET_X = 10
LABEL_OFFSET_Y = 8
CONTROL_MARKER_LONGITUDINAL_SCALE = 0.22
CONTROL_MARKER_LATERAL_OFFSET_SCALE = 0.9
CONTROL_MARKER_DEFLECTION_LENGTH_PX = 14.0
CONTROL_MARKER_RADIUS_PX = 3


class ForceOverlayRenderer:
    """Draw force vectors and a compact physics debug panel."""

    def __init__(
        self,
        viewport: Viewport,
        camera: SceneCamera,
        dynamics: BoosterDynamicsModel,
        *,
        scale_px_per_kn: float = FORCE_VECTOR_SCALE_PX_PER_KN,
    ) -> None:
        self._viewport = viewport
        self._camera = camera
        self._dynamics = dynamics
        self._scale_px_per_kn = scale_px_per_kn
        self._font: pygame.font.Font | None = None
        self._font_small: pygame.font.Font | None = None

    def draw(self, surface: pygame.Surface, state: State, action: Action) -> None:
        """Render vectors, axes and a debug panel for the current frame."""

        self._ensure_fonts()

        params = self._dynamics.params
        mass = self._dynamics.current_mass(state)
        safe_action = self._dynamics.sanitize_action(action)
        thrust = self._dynamics.thrust_for(state, safe_action)
        forces = self._dynamics.forces_for(state, safe_action)
        aerodynamic_loads = self._dynamics.aerodynamic_loads_for(state, safe_action)
        engine_torque = self._dynamics.engine_torque_for(safe_action, thrust)
        torque = engine_torque + aerodynamic_loads.total_torque
        inertia = self._dynamics.moment_of_inertia_for(mass)
        angular_acceleration = torque / inertia if inertia > 0.0 else 0.0

        origin = self._camera.world_to_screen((state.x, state.z + (params.length * 0.5)))
        entries = (
            ("THR", forces.engine, self._viewport.accent),
            ("GRV", forces.gravity, self._viewport.warning),
            ("AERO", forces.aerodynamic, AERODYNAMIC_VECTOR_COLOR),
            ("NET", forces.total, self._viewport.accent_warm),
        )

        self._draw_body_axis(surface, origin, state.theta, params.length)
        self._draw_origin_marker(surface, origin)
        self._draw_aero_control_markers(surface, origin, state.theta, safe_action.aero_steer)
        for label, force, color in entries:
            self._draw_force_vector(surface, origin, label, force, color)
        self._draw_panel(
            surface,
            state,
            safe_action,
            mass,
            thrust,
            forces,
            aerodynamic_loads,
            engine_torque,
            torque,
            angular_acceleration,
        )

    def _ensure_fonts(self) -> None:
        if self._font is None:
            self._font = pygame.font.Font(None, 23)
        if self._font_small is None:
            self._font_small = pygame.font.Font(None, 20)

    def _draw_body_axis(
        self,
        surface: pygame.Surface,
        origin: tuple[int, int],
        theta: float,
        rocket_length_m: float,
    ) -> None:
        half_length_px = max(
            16.0,
            rocket_length_m * self._camera.pixels_per_meter * BODY_AXIS_LENGTH_SCALE * 0.5,
        )
        dx = math.sin(theta) * half_length_px
        dy = math.cos(theta) * half_length_px
        nose = (int(round(origin[0] + dx)), int(round(origin[1] - dy)))
        tail = (int(round(origin[0] - dx)), int(round(origin[1] + dy)))
        pygame.draw.line(surface, BODY_AXIS_COLOR, tail, nose, BODY_AXIS_WIDTH)

    def _draw_origin_marker(self, surface: pygame.Surface, origin: tuple[int, int]) -> None:
        pygame.draw.circle(surface, BODY_AXIS_COLOR, origin, ORIGIN_RADIUS_PX, 2)

    def _draw_aero_control_markers(
        self,
        surface: pygame.Surface,
        origin: tuple[int, int],
        theta: float,
        aero_steer: float,
    ) -> None:
        axis_x, axis_z = self._dynamics.body_axis_for(theta)
        normal_x, normal_z = self._dynamics.body_normal_for(theta)
        longitudinal_offset = max(
            14.0,
            self._dynamics.params.length
            * self._camera.pixels_per_meter
            * CONTROL_MARKER_LONGITUDINAL_SCALE,
        )
        lateral_offset = max(
            6.0,
            self._dynamics.params.radius
            * self._camera.pixels_per_meter
            * CONTROL_MARKER_LATERAL_OFFSET_SCALE,
        )
        base_x = origin[0] + (axis_x * longitudinal_offset)
        base_y = origin[1] - (axis_z * longitudinal_offset)
        left_marker = (
            int(round(base_x + (normal_x * lateral_offset))),
            int(round(base_y - (normal_z * lateral_offset))),
        )
        right_marker = (
            int(round(base_x - (normal_x * lateral_offset))),
            int(round(base_y + (normal_z * lateral_offset))),
        )
        pygame.draw.circle(surface, CONTROL_MARKER_COLOR, left_marker, CONTROL_MARKER_RADIUS_PX)
        pygame.draw.circle(surface, CONTROL_MARKER_COLOR, right_marker, CONTROL_MARKER_RADIUS_PX)

        deflection = aero_steer * CONTROL_MARKER_DEFLECTION_LENGTH_PX
        left_tip = (
            int(round(left_marker[0] + (normal_x * deflection))),
            int(round(left_marker[1] - (normal_z * deflection))),
        )
        right_tip = (
            int(round(right_marker[0] + (normal_x * deflection))),
            int(round(right_marker[1] - (normal_z * deflection))),
        )
        pygame.draw.line(surface, CONTROL_MARKER_COLOR, left_marker, left_tip, 2)
        pygame.draw.line(surface, CONTROL_MARKER_COLOR, right_marker, right_tip, 2)

    def _draw_force_vector(
        self,
        surface: pygame.Surface,
        origin: tuple[int, int],
        label: str,
        force: ForceVector,
        color: tuple[int, int, int],
    ) -> None:
        magnitude = math.hypot(force.x, force.z)
        if magnitude < FORCE_MIN_MAGNITUDE_N:
            return

        target_length = max(
            FORCE_VECTOR_MIN_LENGTH_PX,
            magnitude * (self._scale_px_per_kn / 1_000.0),
        )
        length = min(FORCE_VECTOR_MAX_LENGTH_PX, target_length)
        scale = length / magnitude

        end_x = origin[0] + (force.x * scale)
        end_y = origin[1] - (force.z * scale)
        end = (int(round(end_x)), int(round(end_y)))

        pygame.draw.line(surface, color, origin, end, FORCE_VECTOR_LINE_WIDTH)
        self._draw_arrow_head(surface, color, origin, (end_x, end_y))
        self._draw_vector_label(surface, label, color, end)

    def _draw_arrow_head(
        self,
        surface: pygame.Surface,
        color: tuple[int, int, int],
        start: tuple[int, int],
        end: tuple[float, float],
    ) -> None:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 0.0:
            return

        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux
        base_x = end[0] - (ux * FORCE_VECTOR_HEAD_LENGTH_PX)
        base_y = end[1] - (uy * FORCE_VECTOR_HEAD_LENGTH_PX)
        left = (
            int(round(base_x + (px * FORCE_VECTOR_HEAD_HALF_WIDTH_PX))),
            int(round(base_y + (py * FORCE_VECTOR_HEAD_HALF_WIDTH_PX))),
        )
        right = (
            int(round(base_x - (px * FORCE_VECTOR_HEAD_HALF_WIDTH_PX))),
            int(round(base_y - (py * FORCE_VECTOR_HEAD_HALF_WIDTH_PX))),
        )
        tip = (int(round(end[0])), int(round(end[1])))
        pygame.draw.polygon(surface, color, (tip, left, right))

    def _draw_vector_label(
        self,
        surface: pygame.Surface,
        label: str,
        color: tuple[int, int, int],
        end: tuple[int, int],
    ) -> None:
        assert self._font_small is not None
        label_surface = self._font_small.render(label, True, color)
        shadow_surface = self._font_small.render(label, True, TEXT_SHADOW_COLOR)
        x = end[0] + LABEL_OFFSET_X
        y = end[1] - LABEL_OFFSET_Y
        surface.blit(shadow_surface, (x + 1, y + 1))
        surface.blit(label_surface, (x, y))

    def _draw_panel(
        self,
        surface: pygame.Surface,
        state: State,
        action: Action,
        mass: float,
        thrust: float,
        forces: ForceBreakdown,
        aerodynamic_loads: AerodynamicLoads,
        engine_torque: float,
        total_torque: float,
        angular_acceleration: float,
    ) -> None:
        assert self._font is not None
        assert self._font_small is not None

        lines = [
            ("DEBUG", "PHYSICS"),
            ("mass", f"{mass:,.0f} kg"),
            ("rho", f"{aerodynamic_loads.density:.3f} kg/m3"),
            ("thrust", f"{thrust / 1_000.0:,.1f} kN"),
            ("twr", f"{thrust / max(1.0, mass * self._dynamics.params.gravity):.2f}"),
            (
                "v rel",
                (
                    f"{aerodynamic_loads.relative_velocity.x:+.1f}"
                    f" / {aerodynamic_loads.relative_velocity.z:+.1f} m/s"
                ),
            ),
            (
                "v body",
                f"{aerodynamic_loads.axial_speed:+.1f}"
                f" / {aerodynamic_loads.lateral_speed:+.1f} m/s",
            ),
            ("acc", f"{forces.total.x / mass:+.2f} / {forces.total.z / mass:+.2f} m/s2"),
            ("torque", f"{total_torque / 1_000.0:+.1f} kN.m"),
            ("eng tq", f"{engine_torque / 1_000.0:+.1f} kN.m"),
            ("pass tq", f"{aerodynamic_loads.passive_torque / 1_000.0:+.1f} kN.m"),
            ("ctrl tq", f"{aerodynamic_loads.control_torque / 1_000.0:+.1f} kN.m"),
            ("damp tq", f"{aerodynamic_loads.damping_torque / 1_000.0:+.1f} kN.m"),
            ("ang acc", f"{angular_acceleration:+.3f} rad/s2"),
            ("theta", f"{math.degrees(state.theta):+.2f} deg"),
            ("aoa", f"{math.degrees(aerodynamic_loads.angle_of_attack):+.2f} deg"),
            ("omega", f"{math.degrees(state.omega):+.2f} deg/s"),
            ("eng gmb", f"{math.degrees(action.engine_gimbal):+.2f} deg"),
            ("aero cmd", f"{action.aero_steer * 100:+.0f}%"),
            ("engine", self._format_force_components(forces.engine)),
            ("gravity", self._format_force_components(forces.gravity)),
            ("aero p", self._format_force_components(aerodynamic_loads.passive_force)),
            ("aero c", self._format_force_components(aerodynamic_loads.control_force)),
            ("aero", self._format_force_components(forces.aerodynamic)),
            ("net", self._format_force_components(forces.total)),
        ]

        panel_height = (FORCE_PANEL_PADDING * 2) + (len(lines) * FORCE_LABEL_LINE_HEIGHT)
        rect = pygame.Rect(
            FORCE_LABEL_MARGIN_X,
            FORCE_LABEL_MARGIN_Y,
            FORCE_PANEL_WIDTH,
            panel_height,
        )
        self._draw_panel_background(surface, rect)

        for index, (label, value) in enumerate(lines):
            y = rect.y + FORCE_PANEL_PADDING + (index * FORCE_LABEL_LINE_HEIGHT)
            label_color = self._viewport.text if label != "DEBUG" else self._viewport.accent_warm
            value_color = self._viewport.accent if label == "DEBUG" else self._viewport.text
            label_surface = self._font_small.render(label.upper(), True, label_color)
            value_surface = self._font_small.render(value, True, value_color)
            surface.blit(label_surface, (rect.x + FORCE_PANEL_PADDING, y))
            surface.blit(value_surface, (rect.x + 122, y))

    def _draw_panel_background(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((*self._viewport.panel_bg, FORCE_PANEL_ALPHA))
        pygame.draw.rect(
            panel,
            (*self._viewport.panel_border, FORCE_PANEL_BORDER_ALPHA),
            panel.get_rect(),
            1,
            border_radius=FORCE_PANEL_RADIUS,
        )
        surface.blit(panel, rect)

    @staticmethod
    def _format_force_components(force: ForceVector) -> str:
        magnitude_kn = math.hypot(force.x, force.z) / 1_000.0
        return (
            f"{force.x / 1_000.0:+.1f}, {force.z / 1_000.0:+.1f} kN"
            f"  |  {magnitude_kn:.1f} kN"
        )
