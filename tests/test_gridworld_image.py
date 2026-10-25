"""Tests for drawing sampled episodes with state IDs or coordinate labels."""

import numpy as np
import pytest

from gym_classics2.envs.abstract import gridworld


@pytest.mark.parametrize("tabular", [True, False])
@pytest.mark.parametrize("state_format", ["id", "tuple", "array"])
def test_image_episode_accepts_state_ids_and_labels(monkeypatch, tabular, state_format):
    env = gridworld.Gridworld("S G", tabular=tabular)
    states = [(0, 0), (1, 0), (2, 0)]
    if state_format == "id":
        states = [np.int64(env.state2id(state)) for state in states]
    elif state_format == "array":
        states = [np.array(state) for state in states]
    episode = [
        (states[0], 1, 0.0, states[1]),
        (states[1], 1, 1.0, states[2]),
    ]
    plotted = {}

    def capture_image(matrix, **kwargs):
        plotted.update(kwargs)

    monkeypatch.setattr(gridworld, "_image", capture_image)
    try:
        env.image(episode=episode)
        np.testing.assert_array_equal(plotted["labels"], [["→", "→", ""]])
    finally:
        env.close()
