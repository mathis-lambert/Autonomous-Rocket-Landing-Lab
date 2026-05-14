# Rocket Landing

2D booster descent and landing simulator focused on physics, control, and
interactive visualization.

## Docs

- [Architecture](ARCHITECTURE.md)
- [Physics model](docs/physics_model.md)

## Install

```bash
uv sync
```

## Run The Simulator

Live session with manual control:

```bash
uv run rocket-landing session --controller manual
```

Live session with the scripted baseline controller:

```bash
uv run rocket-landing session --controller baseline
```

Run another YAML scenario:

```bash
uv run rocket-landing session --controller baseline --config configs/offset_recovery.yaml
```

Replay a constant-action scenario:

```bash
uv run rocket-landing demo --throttle 0.85 --gimbal 0.02 --aero-steer 0.0
```

Export a static trajectory plot:

```bash
uv run rocket-landing demo --render-mode plot --output runs/manual_demo.png
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
    control/
    services/
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
- `initial_state`
