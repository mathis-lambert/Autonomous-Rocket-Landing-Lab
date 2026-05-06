# Architecture

This document gives a high-level view of the project structure, the
responsibilities of each layer, and the main data flow through the simulator.

## Goal

The project aims to build a 2D booster descent and landing simulator with a
clean foundation for:

- manual control
- deterministic controllers such as a baseline controller or PID
- reinforcement learning later through `Gymnasium` and `PyTorch`

The current architecture follows a simple three-layer split:

- `domain`
- `application`
- `infrastructure`

## Overview

```text
CLI / Pygame / Matplotlib
            |
            v
application.use_cases + application.control
            |
            v
domain.simulation -> domain.physics -> domain.models
```

The key idea is:

- `domain` knows nothing about `pygame`, `matplotlib`, or the CLI
- `application` orchestrates use cases on top of the domain
- `infrastructure` plugs concrete interfaces around the core

## Tree

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
```

## `domain` layer

The `domain` layer contains the pure simulation model.

### `domain.models`

Contains the simulator value objects.

- `Action`
  Represents a command applied during one simulation step.
  Main fields: `throttle`, `gimbal`

- `State`
  Represents the continuous booster state in the 2D plane.
  Main fields: `x`, `z`, `vx`, `vz`, `theta`, `omega`, `fuel`

- `RocketParams`
  Groups physical constants and landing thresholds.

- `ForceVector`, `StepResult`
  Represent intermediate or output values.

### `domain.physics`

Contains the low-level physics model.

- `dynamics.py`
  Transforms `state + action + params` into linear and angular accelerations,
  plus fuel consumption.

- `integrators.py`
  Advances the state in time using a semi-implicit Euler integrator.

- `collision.py`
  Decides whether ground contact is a `landing` or a `crash`.

- `geometry.py`
  Provides geometry primitives useful for rendering and analysis.

### `domain.simulation`

Orchestrates simulation execution.

- `SimulationEngine`
  Coordinates `dynamics`, `integrator`, and `collision`.

- `SimulationWorld`
  Owns the mutable world state and the fixed time step.

- `SimulationHistory`
  Stores the trajectory over time for analysis or rendering.

## `application` layer

The `application` layer orchestrates concrete use cases.

### `application.control`

Contains controller implementations.

- `FlightController`
  Common abstract interface for any controller.

- `BaselineLandingController`
  A simple deterministic controller used as a baseline before RL.

Later, this layer can naturally host:

- a PID controller
- safety filters
- possibly an RL policy wrapped behind the same interface

### `application.use_cases`

Contains explicit project use cases.

- `RunConstantAction`
  Runs a simulation where the same action is applied at every step.

- `ControlledSimulationSession`
  A mutable session designed for realtime interactive loops.

### `application.services`

Contains non-domain support services.

- `runtime.py`
  Detects the available acceleration backend for future ML workloads:
  `cuda`, `mps`, or `cpu`

## `infrastructure` layer

The `infrastructure` layer adapts the project to the outside world.

### `infrastructure.cli`

- `main.py`
  Main CLI entrypoint

Current subcommands:

- `demo`
- `session`

### `infrastructure.rendering.matplotlib`

- `MatplotlibTrajectoryPlotter`
  Renders a static trajectory for analysis or export.

### `infrastructure.rendering.pygame`

Contains the interactive realtime client.

- `PygameReplayApp`
  Replays an already simulated history.

- `PygameLiveSimulationApp`
  Connects one or more controllers to a live session.

- `PygameReplayScene`
  Draws the world.

- `HeadsUpDisplay`
  Draws telemetry and gauges.

- `SceneCamera`
  Converts world coordinates into screen coordinates.

- `SpriteAssetLoader`
  Loads and preprocesses sprites.

- `PygameKeyboardManualController`
  Keyboard controller implementing the `FlightController` interface.

## Main flow of a live session

The main flow of an interactive `pygame` session is:

```text
PygameLiveSimulationApp
  -> reads keyboard events
  -> lets the controller produce an Action
  -> calls ControlledSimulationSession.step(action)
  -> which calls SimulationWorld.step(action)
  -> which calls SimulationEngine.step(state, action, dt)
  -> which chains:
       BoosterDynamicsModel.evaluate(...)
       SemiImplicitEulerIntegrator.integrate(...)
       GroundContactResolver.resolve(...)
  -> then the renderer reads the history and current state
```

## Flow of a `demo`

The flow of a constant-action `demo` is simpler:

```text
CLI
  -> RunConstantAction.execute(...)
  -> SimulationWorld.step(...)
  -> accumulation into SimulationHistory
  -> rendering through PygameReplayApp or MatplotlibTrajectoryPlotter
```

## Why this architecture

This structure was chosen to preserve:

- a testable simulation core
- rendering decoupled from physics
- gradual growth in complexity
- future RL integration without rewriting the engine

In practice, it allows us to:

- test the physics with `pytest`
- change renderers without breaking the core
- attach multiple controllers to the same simulation
- prepare a clean RL wrapper later

## Design invariants

The following points should remain true as the project evolves:

- physics must never depend on `pygame`
- the `domain` layer must remain importable without UI dependencies
- controllers should expose a consistent interface
- rendering consumes states and histories, but does not drive physics
- the CLI remains a thin assembly layer over use cases

## Natural extensions

The current structure supports the following future additions naturally:

- `application/control/pid.py`
- `application/control/safety_filter.py`
- `application/use_cases/run_training_session.py`
- `infrastructure/rl/gymnasium_env.py`
- `domain/physics/aerodynamics.py`
- `domain/physics/actuators.py`

## Recommended reading order

To understand the project in the right order:

1. `src/rocket_landing/domain/models`
2. `src/rocket_landing/domain/physics`
3. `src/rocket_landing/domain/simulation`
4. `src/rocket_landing/application/use_cases/run_controlled_session.py`
5. `src/rocket_landing/application/control/baseline.py`
6. `src/rocket_landing/infrastructure/rendering/pygame`
