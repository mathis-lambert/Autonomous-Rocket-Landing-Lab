from pathlib import Path

import pytest

from rocket_landing.infrastructure.config.yaml_loader import (
    DEFAULT_CONFIG_PATH,
    load_simulation_config,
)


def test_load_default_simulation_config() -> None:
    config = load_simulation_config()

    assert config.name == "default_descent"
    assert config.params.gravity == 9.81
    assert config.params.max_gimbal == 0.30
    assert config.params.max_landing_x == 5.0
    assert config.params.axial_drag_coefficient == 0.35
    assert config.params.side_drag_coefficient == 1.15
    assert config.params.control_surface_force_coefficient == 0.75
    assert config.initial_state.z == 120.0
    assert config.initial_state.fuel == 12_000.0


def test_load_custom_simulation_config(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: test_case",
                "vehicle:",
                "  dry_mass: 100.0",
                "  initial_fuel: 25.0",
                "  max_thrust: 5000.0",
                "  fuel_flow_rate: 3.0",
                "  length: 6.0",
                "  radius: 0.8",
                "  max_gimbal: 0.2",
                "environment:",
                "  gravity: 1.62",
                "aerodynamics:",
                "  air_density_sea_level: 0.02",
                "  atmosphere_scale_height: 12000.0",
                "  axial_drag_coefficient: 0.4",
                "  side_drag_coefficient: 1.3",
                "  center_of_pressure_offset: 2.5",
                "  angular_damping_coefficient: 0.07",
                "  control_surface_force_coefficient: 0.25",
                "landing:",
                "  max_landing_vz: 2.5",
                "  max_landing_vx: 1.0",
                "  max_landing_theta: 0.05",
                "  max_landing_omega: 0.1",
                "  max_landing_x: 3.0",
                "initial_state:",
                "  x: 10.0",
                "  z: 250.0",
                "  vx: -4.0",
                "  vz: -20.0",
                "  theta: 0.03",
                "  omega: -0.01",
                "  fuel: 18.0",
            ]
        ),
        encoding="utf-8",
    )

    config = load_simulation_config(config_path)

    assert config.name == "test_case"
    assert config.params.max_thrust == 5_000.0
    assert config.params.gravity == 1.62
    assert config.params.axial_drag_coefficient == 0.4
    assert config.params.center_of_pressure_offset == 2.5
    assert config.params.control_surface_force_coefficient == 0.25
    assert config.params.max_landing_x == 3.0
    assert config.initial_state.x == 10.0
    assert config.initial_state.fuel == 18.0


def test_reject_unknown_config_key(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: invalid_case",
                "vehicle:",
                "  dry_mass: 22000.0",
                "  initial_fuel: 8000.0",
                "  max_thrust: 650000.0",
                "  fuel_flow_rate: 120.0",
                "  length: 24.0",
                "  radius: 1.8",
                "  max_gimbal: 0.30",
                "environment:",
                "  gravity: 9.81",
                "aerodynamics:",
                "  air_density_sea_level: 1.225",
                "  atmosphere_scale_height: 8500.0",
                "  axial_drag_coefficient: 0.35",
                "  side_drag_coefficient: 1.15",
                "  center_of_pressure_offset: 7.0",
                "  angular_damping_coefficient: 0.12",
                "  control_surface_force_coefficient: 0.75",
                "landing:",
                "  max_landing_vz: 3.0",
                "  max_landing_vx: 1.5",
                "  max_landing_theta: 0.10",
                "  max_landing_omega: 0.25",
                "  max_landing_x: 5.0",
                "initial_state:",
                "  x: 0.0",
                "  z: 120.0",
                "  vx: 0.0",
                "  vz: -15.0",
                "  theta: 0.0",
                "  omega: 0.0",
                "  fuel: 8000.0",
                "unexpected: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys in config: unexpected"):
        load_simulation_config(config_path)


def test_default_config_path_points_to_existing_file() -> None:
    assert DEFAULT_CONFIG_PATH.exists()
