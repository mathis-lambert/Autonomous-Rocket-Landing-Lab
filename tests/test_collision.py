from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.collision import GroundContactResolver


def test_ground_contact_soft_landing_is_marked_as_landed() -> None:
    params = RocketParams()
    resolver = GroundContactResolver(params)
    state = State(x=0.0, z=-0.5, vx=0.2, vz=-1.0, theta=0.01, omega=0.02, fuel=100.0)

    result = resolver.resolve(state)

    assert result.terminated is True
    assert result.landed is True
    assert result.crashed is False
    assert result.state.z == 0.0


def test_ground_contact_hard_landing_is_marked_as_crashed() -> None:
    params = RocketParams()
    resolver = GroundContactResolver(params)
    state = State(x=0.0, z=-0.5, vx=3.0, vz=-12.0, theta=0.4, omega=1.0, fuel=100.0)

    result = resolver.resolve(state)

    assert result.terminated is True
    assert result.landed is False
    assert result.crashed is True
