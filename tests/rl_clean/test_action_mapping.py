import numpy as np
import pytest

from rocket_landing.domain.models.params import RocketParams
from rocket_landing.rl.env import decode_action


def test_decode_action_uses_symmetric_input_range() -> None:
    params = RocketParams(max_gimbal=0.2)
    action = decode_action(np.array([-1.0, 1.0], dtype=np.float32), params)
    assert action.throttle == pytest.approx(0.0)
    assert action.engine_gimbal == pytest.approx(0.2)

    action_mid = decode_action(np.array([0.0, 0.0], dtype=np.float32), params)
    assert action_mid.throttle == pytest.approx(0.5)
    assert action_mid.engine_gimbal == pytest.approx(0.0)
