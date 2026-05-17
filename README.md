# Rocket Landing

2D booster descent and landing simulator focused on a configurable physics
engine, precise manual control, and realistic live visualization.

## Docs

- [Architecture](ARCHITECTURE.md)
- [Physics model](docs/physics_model.md)

## Install

```bash
uv sync
uv sync --group rl
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

## RL Training

Train SAC with the rebuilt, stage-aligned pipeline:

```bash
uv run python scripts/train_rl.py \
  --run-name curriculum_guard_v2 \
  --timesteps 1500000 \
  --segment-timesteps 25000 \
  --eval-episodes 50 \
  --benchmark-episodes 200 \
  --n-envs 4 \
  --seeds 0 \
  --benchmark-stage full_envelope
```

Evaluate one saved model on a specific stage:

```bash
uv run python scripts/eval_rl.py artifacts/rl/my_run_seed0/models/final_model.zip --stage full_envelope --episodes 100
```

The default curriculum now bridges into the final envelope through
`wide_recovery` and `full_envelope_nominal_fuel` before training on the low-fuel
`full_envelope` distribution. Evaluation reports include truncation reason
rates so failed stages can be diagnosed without replaying every episode.

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
