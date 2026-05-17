from rocket_landing.domain.models.params import RocketParams
from rocket_landing.domain.models.state import State
from rocket_landing.domain.physics.collision import GroundContactResolver


def test_ground_clearance_is_zero_after_ground_contact_snap() -> None:
    params = RocketParams()
    resolver = GroundContactResolver(params)
    descending_state = State(
        x=0.0,
        z=(params.length * 0.5) - 1.0,
        vx=0.0,
        vz=-1.0,
        theta=0.0,
        omega=0.0,
        fuel=params.initial_fuel,
    )

    result = resolver.resolve(descending_state)

    assert result.terminated is True
    assert result.state.z == params.length * 0.5
    assert result.state.ground_clearance(params.length) == 0.0
