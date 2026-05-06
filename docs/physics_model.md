# Physics Model

This document describes the current physical model used by the simulator, the
conventions it relies on, the main equations, and the termination criteria.

The goal is not to reproduce full aerospace flight dynamics, but to provide a
2D model that is coherent, readable, and realistic enough to:

- learn the simulation -> control -> RL pipeline
- debug a hand-written physics engine
- serve as a foundation for more advanced extensions

## Scope of the current model

The model currently represents:

- a rigid booster in a 2D plane
- gravity
- a main engine thrust
- engine gimbal
- rigid-body rotation
- variable mass through fuel burn
- simple ground contact with no rebound

The model does not yet represent:

- aerodynamic drag
- wind
- moving center of mass
- actuator lag
- noisy sensors
- 3D / 6DOF dynamics

## Conventions

### Reference frame

The world uses a 2D frame:

- `x` increases to the right
- `z` increases upward

The ground is defined by:

- `z = 0`

### Attitude

Booster attitude is described by:

- `theta`

Convention:

- `theta = 0`: booster perfectly vertical
- `theta > 0`: top of the booster leans to the right
- `theta < 0`: top of the booster leans to the left

Angular velocity is:

- `omega`

with:

- `omega > 0`: rotation in the positive angle direction

### Commands

An action contains:

- `throttle` in `[0, 1]`
- `gimbal` in radians

Gimbal convention:

- `gimbal = 0`: thrust aligned with the booster axis
- `gimbal > 0`: thrust deflected in the positive angular direction

The effective thrust angle is therefore:

```text
force_angle = theta + gimbal
```

## Simulated state

The booster state is:

```text
x       horizontal position
z       altitude
vx      horizontal velocity
vz      vertical velocity
theta   booster angle
omega   angular velocity
fuel    remaining fuel
```

All units are SI:

- meters
- seconds
- kilograms
- radians

## Physical parameters

The main parameters live in `RocketParams`:

- `gravity`
- `dry_mass`
- `initial_fuel`
- `max_thrust`
- `fuel_flow_rate`
- `length`
- `radius`
- `max_gimbal`

They define both the mechanical envelope and the landing/crash thresholds.

## Mass

Current mass is computed as:

```text
mass = dry_mass + fuel
```

The model does not yet handle internal fuel redistribution. All propellant mass
is effectively treated as if it does not move the center of mass.

## Thrust

Scalar thrust is:

```text
thrust = throttle * max_thrust
```

If fuel is exhausted:

```text
thrust = 0
```

## Engine force

Thrust is projected into the world frame as:

```text
Fx_engine = thrust * sin(theta + gimbal)
Fz_engine = thrust * cos(theta + gimbal)
```

Important consequence:

- if `theta > 0` and `gimbal = 0`, thrust has a rightward horizontal component
- if `theta < 0`, it pushes to the left

## Gravity

Gravity force is:

```text
Fx_gravity = 0
Fz_gravity = -mass * gravity
```

## Total force

The current force sum is:

```text
Fx_total = Fx_engine
Fz_total = Fz_engine - mass * gravity
```

Later, this can be extended with:

- drag
- wind
- external disturbances

## Linear acceleration

Linear acceleration is:

```text
ax = Fx_total / mass
az = Fz_total / mass
```

## Engine torque

The gimbaled engine produces a simplified torque:

```text
torque = lever_arm * thrust * sin(gimbal)
```

with:

```text
lever_arm = length / 2
```

This approximation assumes:

- a main engine located below the center of mass
- a booster approximated as a rigid slender body

## Moment of inertia

The moment of inertia is approximated as that of a thin rod:

```text
I = mass * length^2 / 12
```

## Angular acceleration

Then:

```text
alpha = torque / I
```

## Numerical integration

The project currently uses:

- semi-implicit Euler integration

Update order:

```text
vx = vx + ax * dt
vz = vz + az * dt

x = x + vx * dt
z = z + vz * dt

omega = omega + alpha * dt
theta = theta + omega * dt
```

This choice is intentional:

- easy to understand
- lightweight to implement
- stable enough for the current version

## Fuel consumption

Fuel burned over one step is:

```text
fuel_burn = throttle * fuel_flow_rate * dt
```

Then:

```text
fuel_next = max(0, fuel - fuel_burn)
```

## Ground contact

Ground contact is triggered as soon as:

```text
z <= 0
```

The current model does not handle rebounds or rich contact deformation.

The solver:

1. snaps `z` back to `0`
2. evaluates whether the impact is a landing or a crash
3. stops translational and rotational dynamics

The final grounded state forces:

- `z = 0`
- `vx = 0`
- `vz = 0`
- `omega = 0`

## Successful landing criteria

A contact is classified as `landing` only if all of the following hold:

```text
abs(vz)    <= max_landing_vz
abs(vx)    <= max_landing_vx
abs(theta) <= max_landing_theta
abs(omega) <= max_landing_omega
```

By default, that means:

- low vertical speed
- low lateral drift
- small attitude error
- low angular rate

If any one of these conditions fails, the impact is classified as:

- `crash`

## Step pipeline

A simulation step follows this chain:

```text
Action
  -> sanitize_action
  -> compute thrust
  -> compute engine force
  -> add gravity
  -> compute torque
  -> compute accelerations
  -> integrate state
  -> update fuel
  -> resolve ground contact
  -> StepResult
```

Concretely, this goes through:

1. `BoosterDynamicsModel.evaluate(...)`
2. `SemiImplicitEulerIntegrator.integrate(...)`
3. `GroundContactResolver.resolve(...)`

## Summary of the mathematical model

```text
mass = dry_mass + fuel

thrust = throttle * max_thrust

force_angle = theta + gimbal

Fx = thrust * sin(force_angle)
Fz = thrust * cos(force_angle) - mass * gravity

ax = Fx / mass
az = Fz / mass

torque = (length / 2) * thrust * sin(gimbal)
I = mass * length^2 / 12
alpha = torque / I

vx <- vx + ax * dt
vz <- vz + az * dt
x  <- x + vx * dt
z  <- z + vz * dt

omega <- omega + alpha * dt
theta <- theta + omega * dt

fuel <- max(0, fuel - throttle * fuel_flow_rate * dt)
```

## Known limitations

The current model is intentionally simple.

Main limitations:

- no aerodynamics
- no wind
- no engine lag
- no dynamic saturation beyond simple command clamping
- no center-of-mass variation
- no friction or rich contact behavior

These simplifications are acceptable at this stage because the main objective is
to understand:

- the core equations
- numerical stability
- controllability
- future RL integration

## Recommended extensions

The most natural next improvements are:

1. aerodynamic drag
2. lateral wind
3. throttle/gimbal lag
4. sensor noise
5. randomized initial conditions
6. a proper PID controller
7. a Gymnasium wrapper

## Where to look in the code

The most important files for the physics model are:

- `src/rocket_landing/domain/models/state.py`
- `src/rocket_landing/domain/models/params.py`
- `src/rocket_landing/domain/physics/dynamics.py`
- `src/rocket_landing/domain/physics/integrators.py`
- `src/rocket_landing/domain/physics/collision.py`
- `src/rocket_landing/domain/simulation/engine.py`
