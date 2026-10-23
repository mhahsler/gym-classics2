import numpy as np
import pytest

from gym_classics2.algorithms.schedules import CosineAnnealingSchedule


def test_cosine_annealing_curve_and_endpoint():
    schedule = CosineAnnealingSchedule(1.0, min_value=0.1, decay_steps=100)

    assert schedule(0) == pytest.approx(1.0)
    assert schedule(25) == pytest.approx(0.1 + 0.45 * (1 + np.sqrt(0.5)))
    assert schedule(50) == pytest.approx(0.55)
    assert schedule(75) == pytest.approx(0.1 + 0.45 * (1 - np.sqrt(0.5)))
    assert schedule(100) == pytest.approx(0.1)
    assert schedule(200) == pytest.approx(0.1)
    assert np.all(np.diff([schedule(t) for t in range(201)]) <= 0)


@pytest.mark.parametrize("decay_steps", [0, 1])
def test_cosine_annealing_short_duration(decay_steps):
    schedule = CosineAnnealingSchedule(1.0, min_value=0.1, decay_steps=decay_steps)

    assert schedule(0) == pytest.approx(1.0)
    assert schedule(1) == pytest.approx(0.1)
    assert schedule(2) == pytest.approx(0.1)
