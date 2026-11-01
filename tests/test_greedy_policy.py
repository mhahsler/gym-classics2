"""Greedy tabular policies support both state representations."""

import gymnasium as gym
import numpy as np
import pytest

from gym_classics2.algorithms.policy import _backup, greedy_policy
from gym_classics2.envs.abstract.gridworld import Gridworld


@pytest.mark.parametrize("tabular", [True, False])
def test_greedy_policy_uses_state_ids_for_values_and_policy(tabular):
    env = Gridworld("S G", tabular=tabular)
    try:
        values = np.array([0.0, 0.5, 1000.0])
        start, middle, _ = env.states()
        right = env.action2id("right")

        # Nonterminal successors bootstrap; terminal successors do not.
        assert _backup(env, 0.9, values, start, right) == pytest.approx(0.45)
        assert _backup(env, 0.9, values, middle, right) == pytest.approx(1.0)

        policy = greedy_policy(values, env, discount=0.9, rng=42)
        assert policy.shape == (3,)
        assert np.issubdtype(policy.dtype, np.integer)
        np.testing.assert_array_equal(policy[:2], [right, right])
        assert env.tabular == tabular
    finally:
        env.close()


@pytest.mark.parametrize("discount", [0.0, 0.9, 1.0])
def test_greedy_policy_matches_between_modes_for_stochastic_model(discount):
    tabular = gym.make("ClassicGridworld-v1", tabular=True).unwrapped
    raw = gym.make("ClassicGridworld-v1", tabular=False).unwrapped
    try:
        values = np.random.default_rng(7).normal(size=len(tabular.states()))
        np.testing.assert_array_equal(
            greedy_policy(values, raw, discount=discount, rng=42),
            greedy_policy(values, tabular, discount=discount, rng=42),
        )
    finally:
        tabular.close()
        raw.close()
