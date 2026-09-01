"""Tests for explicitly supplied linear-approximation features."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from gym_classics2.algorithms.linear_approximation import (
    q_hat,
    semi_gradient_Sarsa_0,
)
from gym_classics2.envs.gym_classics2.linear_walks import Walk5


def test_q_hat_uses_the_supplied_state_features():
    env = SimpleNamespace(action_space=gym.spaces.Discrete(2))
    weights = np.arange(5)

    def linear_features(state, env):
        return np.array([1, state])

    def constant_features(state, env):
        return np.array([1, 10])

    assert q_hat(2, 1, weights, env, linear_features) == 4
    assert q_hat(2, 1, weights, env, constant_features) == 20


def test_semi_gradient_sarsa_passes_features_through_the_algorithm():
    env = Walk5()
    feature_calls = 0

    def state_features(state, env):
        nonlocal feature_calls
        feature_calls += 1
        return np.array([1, state])

    weights = semi_gradient_Sarsa_0(
        env,
        state_features,
        n=2,
        epsilon=0.1,
        alpha=0.01,
        gamma=0.9,
        rng=42,
    )

    assert feature_calls > 0
    assert weights.shape == (5,)
