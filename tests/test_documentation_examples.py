"""Smoke tests for examples shown in the README and documentation."""

import numpy as np
import pytest

import gymnasium as gym
import gym_classics2
from gym_classics2.algorithms.dynamic_programming import value_iteration


@pytest.fixture(scope="module", autouse=True)
def registered_environments():
    """Register once because the package intentionally ignores later calls."""
    gym_classics2.register()


def test_readme_quick_start():
    env = gym.make("ClassicGridworld-v1", tabular=True)
    state, info = env.reset(seed=42)

    assert env.observation_space.contains(state)
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(next_state)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()


def test_model_access_example():
    env = gym.make("ClassicGridworld-v1", tabular=True).unwrapped

    assert env.start_states == ((0, 0),)
    assert env.goal_states == ((3, 1), (3, 2))

    state = env.state2id((0, 0))
    action = env.action2id("up")
    next_states, rewards, terminals, probabilities = env.model(state, action)

    assert next_states == [0, 1, 3]
    np.testing.assert_allclose(rewards, [-0.04, -0.04, -0.04])
    np.testing.assert_array_equal(terminals, [False, False, False])
    np.testing.assert_allclose(probabilities, [0.1, 0.8, 0.1])


def test_dynamic_programming_tutorial():
    env = gym.make("ClassicGridworld-v1", tabular=True).unwrapped
    values = value_iteration(env, discount=0.99, precision=1e-6)

    assert values.shape == (len(env.states()),)
    assert np.isfinite(values).all()


@pytest.mark.parametrize("env_id", ["5Walk-v0", "19Walk-v0"])
def test_documented_random_walks(env_id):
    env = gym.make(env_id)
    state, _ = env.reset(seed=42)
    next_state, reward, terminated, truncated, _ = env.step(
        env.action_space.sample()
    )

    assert env.observation_space.contains(state)
    assert env.observation_space.contains(next_state)
    assert isinstance(terminated, bool)
    assert truncated is False
    env.close()
