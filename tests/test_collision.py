from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.collision import GroundContactResolver


def test_ground_contact_soft_landing_is_marked_as_landed() -> None:
    params = RocketParams()
    resolver = GroundContactResolver(params)
    state = State(
        x=0.0,
        z=(params.length / 2.0) - 0.5,
        vx=0.2,
        vz=-1.0,
        theta=0.01,
        omega=0.02,
        fuel=100.0,
    )

    result = resolver.resolve(state)

    assert result.terminated is True
    assert result.landed is True
    assert result.crashed is False
    assert result.state.z > 0.0
    assert result.impact_state is not None
    assert result.impact_state.vz == -1.0


def test_ground_contact_hard_landing_is_marked_as_crashed() -> None:
    params = RocketParams()
    resolver = GroundContactResolver(params)
    state = State(
        x=0.0,
        z=10.5,
        vx=3.0,
        vz=-12.0,
        theta=0.4,
        omega=1.0,
        fuel=100.0,
    )

    result = resolver.resolve(state)

    assert result.terminated is True
    assert result.landed is False
    assert result.crashed is True
    assert result.impact_state is not None
    assert result.impact_state.vz == -12.0


def test_ground_contact_outside_pad_is_marked_as_crashed() -> None:
    params = RocketParams(max_landing_x=5.0)
    resolver = GroundContactResolver(params)
    state = State(
        x=25.0,
        z=(params.length / 2.0) - 0.5,
        vx=0.2,
        vz=-1.0,
        theta=0.01,
        omega=0.02,
        fuel=100.0,
    )

    result = resolver.resolve(state)

    assert result.terminated is True
    assert result.landed is False
    assert result.crashed is True


def test_ground_contact_triggers_when_bottom_crosses_ground_before_center() -> None:
    params = RocketParams(length=24.0)
    resolver = GroundContactResolver(params)
    state = State(x=0.0, z=11.9, vx=0.0, vz=-1.0, theta=0.0, omega=0.0, fuel=100.0)

    result = resolver.resolve(state)

    assert result.terminated is True
    assert result.impact_state is not None
    assert result.impact_state.z == params.length / 2.0
