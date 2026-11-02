"""Approximation errors stop at the calculation and preserve caller settings."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from gym_classics2.algorithms.eligibility_traces import semi_gradient_Sarsa_lambda
from gym_classics2.algorithms.linear_approximation import (
    MSVE,
    semi_gradient_Sarsa0,
    semi_gradient_TD0,
    transformation_fourier_basis,
)
from gym_classics2.algorithms.policy_gradient_methods import AC, REINFORCE, pi


@pytest.mark.filterwarnings("ignore:The environment has a discrete state space")
@pytest.mark.parametrize(
    "learner, options",
    [
        (semi_gradient_TD0, {"policy": np.array([0])}),
        (semi_gradient_Sarsa0, {"epsilon": 0}),
        (semi_gradient_Sarsa_lambda, {"epsilon": 0, "lam": 0.5}),
        (REINFORCE, {}),
        (AC, {}),
    ],
)
@pytest.mark.parametrize("overflow", [False, True])
def test_training_error_settings(learner, options, overflow):
    reward = 1e200 if overflow else 1.0
    alpha = 1e200 if overflow else 0.1
    env = SimpleNamespace(
        observation_space=gym.spaces.Discrete(1),
        action_space=gym.spaces.Discrete(1),
        reset=lambda: (0, {}),
        step=lambda action: (0, reward, True, False, {}),
    )

    def features(state, env):
        return np.array([1.0])

    step_sizes = {"alpha_policy": alpha, "alpha_value": alpha} if learner is AC else {"alpha": alpha}
    with np.errstate(over="ignore", invalid="ignore"):
        original_settings = np.geterr()
        if overflow:
            with pytest.raises(FloatingPointError, match="overflow"):
                learner(env, features, n=2, gamma=0.9, **step_sizes, **options)
        else:
            result = learner(env, features, n=2, gamma=0.9, **step_sizes, **options)
            parameters = result if learner is AC else (result,)
            assert all(np.isfinite(values).all() for values in parameters)
        assert np.geterr() == original_settings


def test_standalone_approximation_helpers_raise_on_overflow():
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(FloatingPointError, match="overflow"):
            MSVE(np.array([1e200]), np.array([0.0]))

        env = SimpleNamespace(action_space=gym.spaces.Discrete(1))
        with pytest.raises(FloatingPointError, match="overflow"):
            pi(0, np.array([1000.0, 0.0]), env, lambda state, env: np.array([1.0]))

        transform = transformation_fourier_basis([-1e308], [1e308], order=1)
        with pytest.raises(FloatingPointError, match="overflow"):
            transform([0.0])
