# Architecture

This project is structured as a small layered simulator.

## Goal

The codebase aims to stay simple:

- a pure simulation core
- a thin application layer for use cases
- a dedicated RL layer for training/evaluation
- infrastructure adapters for CLI and rendering

## Overview

```text
CLI / Pygame / Matplotlib
          |
          v
application.use_cases + rl
          |
          v
domain.simulation -> domain.physics -> domain.models
```

## Tree

```text
src/rocket_landing/
  domain/
    models/
    physics/
    simulation/
  rl/
  application/
    use_cases/
  infrastructure/
    cli/
    rendering/
```

## Domain

The `domain` layer contains the simulation model.

- `domain.models`
  State, actions, parameters, force vectors, step results.
- `domain.physics`
  Dynamics, aerodynamics, integration, geometry, and ground contact.
- `domain.simulation`
  Engine, world state, and trajectory history.

This layer does not depend on `pygame`, `matplotlib`, or the CLI.

## Application

The `application` layer orchestrates use cases around the simulator.

- `application.use_cases`
  Constant-action runs and live controlled sessions.

## RL

The `rl` layer contains the policy training and evaluation stack.

- `rl.env`
  Gymnasium environment around the simulator.
- `rl.reward`
  Reward shaping and terminal summaries, including guidance terms for high
  altitude lateral recovery.
- `rl.curriculum`
  Progressive stage configuration and promotion rules. Promotion checks success,
  crashes, truncations, and terminal constraint metrics before moving forward.
- `rl.sb3`
  SAC training loop, checkpoints, and evaluation reports.

## Infrastructure

The `infrastructure` layer adapts the simulator to concrete tools.

- `infrastructure.cli`
  Command-line entrypoint.
- `infrastructure.rendering.matplotlib`
  Static trajectory plots.
- `infrastructure.rendering.pygame`
  Live view, replay view, camera, HUD, and sprite rendering.
- `infrastructure.config`
  YAML loading and scenario resolution.

## Main Flow

Interactive session:

```text
PygameLiveSimulationApp
  -> controller.compute_action(...)
  -> ControlledSimulationSession.step(...)
  -> SimulationWorld.step(...)
  -> SimulationEngine.step(...)
  -> dynamics + integrator + collision
```

Replay flow:

```text
RunConstantAction / ControlledSimulationSession
  -> SimulationHistory
  -> PygameReplayApp or MatplotlibTrajectoryPlotter
```
