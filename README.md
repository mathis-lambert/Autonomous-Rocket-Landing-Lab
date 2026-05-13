# Rocket Landing

Personal Python project focused on building a credible rocket flight simulator,
then using it as a training environment for reinforcement learning.

The long-term goal is to study autonomous powered landing on a reusable launch
vehicle, with a project direction inspired by SpaceX flight profiles and control
challenges.

## Vision

This repository is intended to grow into a complete experimentation stack with
two tightly connected parts:

- a flight simulation core that models rocket dynamics, control inputs, and
  landing constraints with enough fidelity to make control problems meaningful
- an RL training environment where agents can learn to stabilize, descend, and
  land the vehicle autonomously

The target is not arcade gameplay. The target is a serious personal simulator
that stays simple where possible, but remains technically useful for:

- manual piloting experiments
- deterministic controller baselines
- reward design and curriculum learning
- policy evaluation and comparison
- future iteration toward more realistic rocket flight behavior

## Project Direction

The final project vision is to support a progression like this:

1. build a robust simulation core with explicit physics and strong automated tests
2. validate the simulator with manual control and baseline autopilots
3. expose the environment to RL tooling
4. train landing policies that can handle increasingly difficult scenarios
5. improve physical realism over time without rewriting the architecture

The simulator is currently 2D and intentionally simplified. That is a design
choice, not an accident: the goal is to establish a clean foundation before
adding more realism such as aerodynamic effects, disturbances, actuator
constraints, richer mission phases, or a more Starship-like control problem.

## Docs

- [Architecture](ARCHITECTURE.md)
- [Physics model](docs/physics_model.md)

## Quick Start

Install dependencies:

```bash
uv sync
```

Run the live session with manual control:

```bash
uv run rocket-landing session --controller manual
```

Run the live session with the baseline controller:

```bash
uv run rocket-landing session --controller baseline
```

Run a tweaked scenario from a YAML file:

```bash
uv run rocket-landing session --controller baseline --config configs/offset_recovery.yaml
```

Replay a constant-action scenario:

```bash
uv run rocket-landing demo --throttle 0.85 --gimbal 0.02 --render-mode replay --playback-speed 1.0
```

Save a trajectory figure instead of opening a window:

```bash
uv run rocket-landing demo --output runs/manual_demo.png
```

Run the constant-action demo without any renderer:

```bash
uv run rocket-landing demo --render-mode none
```

Run the test suite:

```bash
uv run pytest
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
```

## Configuration

Simulation scenarios live in `configs/` and are defined in YAML.

Each scenario currently separates:

- `vehicle`
- `environment`
- `landing`
- `initial_state`

The CLI loads `configs/default.yaml` when no explicit `--config` is provided.
