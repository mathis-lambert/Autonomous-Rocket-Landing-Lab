from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.infrastructure.rendering.pygame.camera import SceneCamera
from rocket_landing.infrastructure.rendering.pygame.viewport import Viewport


def test_camera_centers_on_low_altitude_vehicle() -> None:
    viewport = Viewport(width=1280, height=720)
    params = RocketParams()
    camera = SceneCamera(params, viewport)
    state = State(x=0.0, z=120.0, vx=0.0, vz=-15.0, theta=0.0, omega=0.0, fuel=8_000.0)

    camera.update(state, dt=1 / 60)

    rocket_center = camera.world_to_screen((state.x, state.z + (params.length * 0.5)))
    assert rocket_center == (viewport.width // 2, viewport.height // 2)


def test_camera_keeps_high_altitude_vehicle_centered_without_auto_zooming() -> None:
    viewport = Viewport(width=1280, height=720)
    params = RocketParams()
    camera = SceneCamera(params, viewport)
    state = State(x=0.0, z=80_000.0, vx=0.0, vz=-1_200.0, theta=0.0, omega=0.0, fuel=8_000.0)

    camera.update(state, dt=1 / 60)

    rocket_center = camera.world_to_screen((state.x, state.z + (params.length * 0.5)))
    assert rocket_center == (viewport.width // 2, viewport.height // 2)
    assert camera.pixels_per_meter == 3.2


def test_camera_manual_zoom_controls_scale() -> None:
    camera = SceneCamera(RocketParams(), Viewport(width=1280, height=720))

    initial_scale = camera.pixels_per_meter
    camera.zoom_out()
    assert camera.pixels_per_meter < initial_scale

    camera.zoom_in()
    assert camera.pixels_per_meter == initial_scale

    camera.zoom_in()
    assert camera.pixels_per_meter > initial_scale

    camera.reset_zoom()
    assert camera.pixels_per_meter == initial_scale
