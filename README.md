# Rocket Landing

Minimal 2D booster landing simulator written in Python with `uv`.

## Docs

- [Architecture](ARCHITECTURE.md)
- [Physics model](docs/physics_model.md)

## Scope

This first milestone focuses on:

- a small rigid-body 2D physics core
- a reproducible CLI demo
- a realtime `pygame` renderer
- Matplotlib trajectory export
- physics-oriented unit tests
- a runtime helper that detects `cuda`, `mps`, or `cpu` for future ML work

## Quick start

```powershell
uv sync
uv run rocket-landing session --controller manual
```

Replay a constant-action scenario:

```powershell
uv run rocket-landing demo --throttle 0.85 --gimbal 0.02 --render-mode replay --playback-speed 1.0
```

Save a figure instead of opening a window:

```powershell
uv run rocket-landing demo --output runs/manual_demo.png
```

Run the live session with the baseline controller:

```powershell
uv run rocket-landing session --controller baseline
```

Run the constant-action demo without any renderer:

```powershell
uv run rocket-landing demo --render-mode none
```

Run the test suite:

```powershell
uv run pytest
```

## Layout

```text
src/rocket_landing/
  domain/
    models/
    physics/
    simulation/
  application/
    services/
    use_cases/
  infrastructure/
    cli/
    rendering/
scripts/
  run_manual.py
  run_freefall.py
tests/
```

## Next steps

- add a PID controller baseline
- wrap the simulator with Gymnasium
- add PPO and evaluation tooling
- introduce drag, wind, and actuator dynamics
