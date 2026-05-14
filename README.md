# Rocket Landing

2D booster descent and landing simulator focused on a configurable physics
engine, precise manual control, and realistic live visualization.

## Docs

- [Architecture](ARCHITECTURE.md)
- [Physics model](docs/physics_model.md)

## Install

```bash
uv sync
```

## Run The Simulator

Launch the live simulator:

```bash
uv run rocket-landing
```

Run another YAML scenario:

```bash
uv run rocket-landing --config configs/offset_recovery.yaml
```

Start with the force debug overlay enabled:

```bash
uv run rocket-landing --debug-forces
```

## Tests

```bash
uv run pytest
uv run ruff check src tests
```

## Repository Layout

```text
src/rocket_landing/
  domain/
    models/
    physics/
    simulation/
  application/
    use_cases/
  infrastructure/
    cli/
    rendering/
tests/
configs/
docs/
```

## Configuration

Scenario files live in `configs/` and are defined in YAML.

Each scenario currently separates:

- `vehicle`
- `environment`
- `aerodynamics`
- `landing`
- `controls`
- `initial_state`
