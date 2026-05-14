"""Load simulation scenarios from YAML files."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.infrastructure.config.models import ManualControlConfig, SimulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"

_ROCKET_PARAM_FIELDS = {field.name for field in fields(RocketParams)}
_STATE_FIELDS = {field.name for field in fields(State)}

_CONFIG_SECTIONS = {
    "name",
    "description",
    "vehicle",
    "environment",
    "aerodynamics",
    "landing",
    "controls",
    "initial_state",
}
_VEHICLE_FIELDS = {
    "dry_mass",
    "initial_fuel",
    "max_thrust",
    "fuel_flow_rate",
    "length",
    "radius",
    "max_gimbal",
}
_ENVIRONMENT_FIELDS = {"gravity"}
_AERODYNAMICS_FIELDS = {
    "air_density_sea_level",
    "atmosphere_scale_height",
    "axial_drag_coefficient",
    "side_drag_coefficient",
    "center_of_pressure_offset",
    "angular_damping_coefficient",
    "control_surface_force_coefficient",
}
_LANDING_FIELDS = {
    "target_x",
    "max_landing_vz",
    "max_landing_vx",
    "max_landing_theta",
    "max_landing_omega",
    "max_landing_x",
}
_CONTROL_FIELDS = {
    "throttle_rate",
    "engine_gimbal_rate",
    "aero_steer_rate",
    "steering_return_rate",
}


def load_simulation_config(config_path: str | Path | None = None) -> SimulationConfig:
    """Load and validate a simulation configuration from YAML.

    Args:
        config_path: Optional path to a scenario file. When omitted, the
            repository default configuration is loaded.
    """

    resolved_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    data = _load_yaml_mapping(resolved_path)
    return _parse_simulation_config(data, source_path=resolved_path)


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Read a YAML file and require a top-level mapping."""

    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"simulation config not found: {path}") from exc

    loaded = yaml.safe_load(raw_text)
    if not isinstance(loaded, dict):
        raise ValueError(f"simulation config must be a YAML mapping: {path}")
    return loaded


def _parse_simulation_config(data: dict[str, Any], *, source_path: Path) -> SimulationConfig:
    """Validate a configuration payload and convert it into domain objects."""

    _validate_unknown_keys("config", data, _CONFIG_SECTIONS)

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"config field 'name' must be a non-empty string: {source_path}")

    description = data.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError(f"config field 'description' must be a string: {source_path}")

    vehicle_data = _require_mapping(data, "vehicle", source_path=source_path)
    environment_data = _require_mapping(data, "environment", source_path=source_path)
    aerodynamics_data = _require_mapping(data, "aerodynamics", source_path=source_path)
    landing_data = _require_mapping(data, "landing", source_path=source_path)
    controls_data = _require_mapping(data, "controls", source_path=source_path)
    initial_state_data = _require_mapping(data, "initial_state", source_path=source_path)

    _validate_unknown_keys("vehicle", vehicle_data, _VEHICLE_FIELDS)
    _validate_unknown_keys("environment", environment_data, _ENVIRONMENT_FIELDS)
    _validate_unknown_keys("aerodynamics", aerodynamics_data, _AERODYNAMICS_FIELDS)
    _validate_unknown_keys("landing", landing_data, _LANDING_FIELDS)
    _validate_unknown_keys("controls", controls_data, _CONTROL_FIELDS)
    _validate_unknown_keys("initial_state", initial_state_data, _STATE_FIELDS)

    params_payload = {
        **vehicle_data,
        **environment_data,
        **aerodynamics_data,
        **landing_data,
    }
    _validate_missing_keys(
        "rocket parameters",
        params_payload,
        _ROCKET_PARAM_FIELDS,
        source_path=source_path,
    )
    _validate_missing_keys(
        "initial_state",
        initial_state_data,
        _STATE_FIELDS,
        source_path=source_path,
    )
    _validate_missing_keys(
        "controls",
        controls_data,
        _CONTROL_FIELDS,
        source_path=source_path,
    )

    params = RocketParams(**params_payload)
    initial_state = State(**initial_state_data)
    controls = ManualControlConfig(**controls_data)
    _validate_physical_ranges(
        params=params,
        initial_state=initial_state,
        controls=controls,
        source_path=source_path,
    )

    return SimulationConfig(
        name=name.strip(),
        description=description.strip() if isinstance(description, str) else None,
        params=params,
        initial_state=initial_state,
        controls=controls,
    )


def _require_mapping(
    data: dict[str, Any],
    key: str,
    *,
    source_path: Path,
) -> dict[str, Any]:
    """Extract a required mapping section from a config payload."""

    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config section '{key}' must be a mapping: {source_path}")
    return value


def _validate_unknown_keys(section_name: str, data: dict[str, Any], allowed_keys: set[str]) -> None:
    """Reject unexpected keys to keep scenario files explicit and debuggable."""

    unknown_keys = sorted(set(data) - allowed_keys)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"unknown keys in {section_name}: {joined}")


def _validate_missing_keys(
    section_name: str,
    data: dict[str, Any],
    required_keys: set[str],
    *,
    source_path: Path,
) -> None:
    """Reject incomplete sections before constructing dataclasses."""

    missing_keys = sorted(required_keys - set(data))
    if missing_keys:
        joined = ", ".join(missing_keys)
        raise ValueError(f"missing keys in {section_name} for {source_path}: {joined}")


def _validate_physical_ranges(
    *,
    params: RocketParams,
    initial_state: State,
    controls: ManualControlConfig,
    source_path: Path,
) -> None:
    """Reject obviously non-physical or unusable runtime configurations."""

    validations = (
        (params.gravity > 0.0, "gravity must be strictly positive"),
        (params.dry_mass > 0.0, "dry_mass must be strictly positive"),
        (params.initial_fuel >= 0.0, "initial_fuel must be non-negative"),
        (params.max_thrust > 0.0, "max_thrust must be strictly positive"),
        (params.fuel_flow_rate >= 0.0, "fuel_flow_rate must be non-negative"),
        (params.length > 0.0, "length must be strictly positive"),
        (params.radius > 0.0, "radius must be strictly positive"),
        (params.max_gimbal >= 0.0, "max_gimbal must be non-negative"),
        (
            params.atmosphere_scale_height > 0.0,
            "atmosphere_scale_height must be strictly positive",
        ),
        (
            params.air_density_sea_level >= 0.0,
            "air_density_sea_level must be non-negative",
        ),
        (
            params.axial_drag_coefficient >= 0.0,
            "axial_drag_coefficient must be non-negative",
        ),
        (
            params.side_drag_coefficient >= 0.0,
            "side_drag_coefficient must be non-negative",
        ),
        (
            params.center_of_pressure_offset >= 0.0,
            "center_of_pressure_offset must be non-negative",
        ),
        (
            params.angular_damping_coefficient >= 0.0,
            "angular_damping_coefficient must be non-negative",
        ),
        (
            params.control_surface_force_coefficient >= 0.0,
            "control_surface_force_coefficient must be non-negative",
        ),
        (params.max_landing_vz >= 0.0, "max_landing_vz must be non-negative"),
        (params.max_landing_vx >= 0.0, "max_landing_vx must be non-negative"),
        (params.max_landing_theta >= 0.0, "max_landing_theta must be non-negative"),
        (params.max_landing_omega >= 0.0, "max_landing_omega must be non-negative"),
        (params.max_landing_x >= 0.0, "max_landing_x must be non-negative"),
        (initial_state.fuel >= 0.0, "initial_state.fuel must be non-negative"),
        (
            initial_state.fuel <= params.initial_fuel,
            "initial_state.fuel must not exceed initial_fuel",
        ),
        (controls.throttle_rate > 0.0, "controls.throttle_rate must be strictly positive"),
        (
            controls.engine_gimbal_rate > 0.0,
            "controls.engine_gimbal_rate must be strictly positive",
        ),
        (controls.aero_steer_rate > 0.0, "controls.aero_steer_rate must be strictly positive"),
        (
            controls.steering_return_rate > 0.0,
            "controls.steering_return_rate must be strictly positive",
        ),
    )
    for is_valid, message in validations:
        if not is_valid:
            raise ValueError(f"{message}: {source_path}")
