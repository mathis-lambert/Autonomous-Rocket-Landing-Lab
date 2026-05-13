from rocket_landing.domain.models.action import Action
from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.dynamics import BoosterDynamicsModel


def test_clamp_action_enforces_bounds() -> None:
    params = RocketParams()
    model = BoosterDynamicsModel(params)
    action = model.sanitize_action(Action(throttle=2.0, gimbal=1.0))

    assert action.throttle == 1.0
    assert action.gimbal == params.max_gimbal


def test_zero_fuel_disables_thrust() -> None:
    params = RocketParams()
    state = State(x=0.0, z=10.0, vx=0.0, vz=0.0, theta=0.0, omega=0.0, fuel=0.0)
    model = BoosterDynamicsModel(params)

    thrust = model.thrust_for(state, Action(throttle=1.0, gimbal=0.0))

    assert thrust == 0.0


def test_vertical_engine_force_has_no_horizontal_component_when_upright() -> None:
    params = RocketParams()
    state = State(x=0.0, z=10.0, vx=0.0, vz=0.0, theta=0.0, omega=0.0, fuel=100.0)
    model = BoosterDynamicsModel(params)
    action = Action(throttle=1.0, gimbal=0.0)
    thrust = model.thrust_for(state, action)

    force = model.engine_force_for(state, action, thrust)

    assert force.x == 0.0
    assert force.z == thrust


def test_gravity_force_points_downward() -> None:
    params = RocketParams()
    state = State(x=0.0, z=10.0, vx=0.0, vz=0.0, theta=0.0, omega=0.0, fuel=100.0)
    model = BoosterDynamicsModel(params)
    mass = model.current_mass(state)

    force = model.gravity_force_for(mass)

    assert force.x == 0.0
    assert force.z < 0.0


def test_atmospheric_density_decreases_with_altitude() -> None:
    model = BoosterDynamicsModel(RocketParams())

    sea_level_density = model.atmospheric_density_for(0.0)
    high_altitude_density = model.atmospheric_density_for(20_000.0)

    assert high_altitude_density < sea_level_density
    assert high_altitude_density > 0.0


def test_drag_force_opposes_velocity() -> None:
    params = RocketParams(axial_drag_coefficient=0.8, side_drag_coefficient=1.2)
    state = State(x=0.0, z=500.0, vx=30.0, vz=-40.0, theta=0.0, omega=0.0, fuel=100.0)
    model = BoosterDynamicsModel(params)

    force = model.aerodynamic_force_for(state)

    assert force.x < 0.0
    assert force.z > 0.0


def test_force_breakdown_sums_engine_and_gravity() -> None:
    params = RocketParams()
    state = State(x=0.0, z=10.0, vx=0.0, vz=0.0, theta=0.0, omega=0.0, fuel=100.0)
    model = BoosterDynamicsModel(params)

    forces = model.forces_for(state, Action(throttle=0.5, gimbal=0.0))

    assert forces.engine.x == 0.0
    assert forces.gravity.x == 0.0
    assert forces.aerodynamic.x == 0.0
    assert forces.aerodynamic.z == 0.0
    assert forces.total.x == 0.0
    assert forces.total.z == forces.engine.z + forces.gravity.z + forces.aerodynamic.z


def test_positive_angle_of_attack_creates_restoring_aerodynamic_torque() -> None:
    params = RocketParams(
        side_drag_coefficient=1.15,
        center_of_pressure_offset=7.0,
        angular_damping_coefficient=0.12,
    )
    state = State(x=0.0, z=1_000.0, vx=10.0, vz=60.0, theta=0.4, omega=0.0, fuel=100.0)
    model = BoosterDynamicsModel(params)

    torque = model.aerodynamic_torque_for(state)

    assert torque < 0.0


def test_positive_omega_creates_negative_aerodynamic_damping_torque() -> None:
    params = RocketParams(
        side_drag_coefficient=1.15,
        center_of_pressure_offset=7.0,
        angular_damping_coefficient=0.12,
    )
    state = State(x=0.0, z=1_000.0, vx=0.0, vz=60.0, theta=0.0, omega=0.4, fuel=100.0)
    model = BoosterDynamicsModel(params)

    torque = model.aerodynamic_torque_for(state)

    assert torque < 0.0


def test_positive_theta_pushes_booster_to_the_right() -> None:
    params = RocketParams()
    state = State(x=0.0, z=10.0, vx=0.0, vz=0.0, theta=0.1, omega=0.0, fuel=100.0)
    model = BoosterDynamicsModel(params)
    action = Action(throttle=1.0, gimbal=0.0)
    thrust = model.thrust_for(state, action)

    force = model.engine_force_for(state, action, thrust)

    assert force.x > 0.0
