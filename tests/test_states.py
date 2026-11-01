"""State enumeration follows the environment's observation representation."""

import numpy as np
import pytest

from gym_classics2.algorithms.policy import make_multidiscrete_policy
from gym_classics2.envs.abstract.gridworld import Gridworld


@pytest.mark.parametrize("tabular", [True, False])
def test_states_match_model_inputs(tabular):
    env = Gridworld("S G", tabular=tabular)
    try:
        states = env.states()
        if tabular:
            assert isinstance(states, range)
            assert list(states) == [0, 1, 2]
        else:
            assert isinstance(states, list)
            assert states == [(0, 0), (1, 0), (2, 0)]

        for state in states:
            for action in env.actions():
                next_states, _, _, _ = env.model(state, action)
                assert all(next_state in states for next_state in next_states)

        np.testing.assert_array_equal(env.to_matrix(), [[0, 1, 2]])
    finally:
        env.close()


def test_raw_states_list_does_not_expose_decoder():
    env = Gridworld("SG", tabular=False)
    try:
        env.states().clear()
        assert env.states() == [(0, 0), (1, 0)]
        assert env.id2state(0) == (0, 0)
        assert make_multidiscrete_policy(np.array([1, 0]), env) == {
            (0, 0): 1, (1, 0): 0,
        }
    finally:
        env.close()
