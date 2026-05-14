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

    return SimulationConfig(
        name=name.strip(),
        description=description.strip() if isinstance(description, str) else None,
        params=RocketParams(**params_payload),
        initial_state=State(**initial_state_data),
        controls=ManualControlConfig(**controls_data),
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
