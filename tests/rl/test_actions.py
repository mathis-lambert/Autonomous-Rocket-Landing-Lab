import pytest

from rocket_landing.application.rl.encoding.actions import decode_agent_action
from rocket_landing.domain.models.params import RocketParams


def test_decode_agent_action_maps_gimbal_and_clips_inputs() -> None:
    params = RocketParams(max_gimbal=0.30)

    action = decode_agent_action([1.4, -1.8], params)

    assert action.throttle == pytest.approx(1.0)
    assert action.engine_gimbal == pytest.approx(-0.30)
    assert action.aero_steer == pytest.approx(0.0)


def test_decode_agent_action_rejects_unexpected_shape() -> None:
    with pytest.raises(ValueError, match="expected RL action shape"):
        decode_agent_action([0.5], RocketParams())
