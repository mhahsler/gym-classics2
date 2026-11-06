"""Reproducibility tests for algorithm-level random number generation."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from gym_classics2.algorithms.policy import epsilon_greedy_action, random_policy
from gym_classics2.algorithms.temporal_difference_learning import Q_learning
from gym_classics2.envs.gym_classics2.linear_walks import Walk5
from gym_classics2.utils import random_argmax


def test_random_helpers_share_a_reproducible_generator_stream():
    env = SimpleNamespace(
        observation_space=gym.spaces.Discrete(20),
        action_space=gym.spaces.Discrete(4),
    )

    def samples(seed):
        rng = np.random.default_rng(seed)
        policy = random_policy(env, rng=rng)
        actions = [
            epsilon_greedy_action(np.ones(4), epsilon=0.5, rng=rng)
            for _ in range(20)
        ]
        ties = random_argmax(np.ones((10, 4)), axis=1, rng=rng)
        return policy, actions, ties

    first = samples(42)
    second = samples(42)

    for actual, expected in zip(first, second):
        np.testing.assert_array_equal(actual, expected)

    np.testing.assert_array_equal(
        random_policy(env, rng=42),
        random_policy(env, rng=42),
    )


def test_q_learning_uses_rng_instead_of_action_space_rng():
    def learn(action_space_seed):
        env = Walk5()
        env.reset(seed=7)
        env.action_space.seed(action_space_seed)
        return Q_learning(
            env,
            discount=0.9,
            alpha=0.2,
            epsilon=0.3,
            n=20,
            rng=np.random.default_rng(123),
        )

    np.testing.assert_array_equal(learn(1), learn(999))


def test_sarsa_lambda_is_reproducible_and_traces_spread_updates():
    from gym_classics2.algorithms.eligibility_traces import Sarsa_lambda

    def run(f, **kw):
        return f(Walk5(), discount=0.9, alpha=0.2, epsilon=0.3, n=20, rng=3, **kw)

    np.testing.assert_array_equal(run(Sarsa_lambda, lambda_=0.8), run(Sarsa_lambda, lambda_=0.8))
    Q, hist = Sarsa_lambda(Walk5(), 0.9, 0.2, 0.3, lambda_=0.8, n=5, history=True, rng=1)
    assert len(hist['returns']) == len(hist['ep_lens']) == 5
    # More traces propagate reward further in one episode than Sarsa(0).
    Q0 = Sarsa_lambda(Walk5(), 1, 0.5, 0.5, lambda_=0, n=1, rng=2)
    Q9 = Sarsa_lambda(Walk5(), 1, 0.5, 0.5, lambda_=0.9, n=1, rng=2)
    assert np.count_nonzero(Q9) >= np.count_nonzero(Q0)
