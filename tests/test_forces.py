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
