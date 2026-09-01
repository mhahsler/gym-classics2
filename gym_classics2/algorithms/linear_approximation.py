"""Linear function approximation algorithms for policy evaluation and control.

The algorithms do not require discrete state spaces. Callers provide a
``state_features(state, env)`` function that converts states to state feature vectors.
"""

import numpy as np
from itertools import product
import warnings
from tqdm import tqdm

import gymnasium as gym

from gym_classics2.utils import get_rng, random_argmax
from gym_classics2.algorithms.policy import random_policy
from gym_classics2.algorithms.schedules import Schedule, ConstantSchedule
from gym_classics2.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

def state_action_features(s, a, env, state_features):
    """Construct the state-action feature vector ``x(s, a)``.

    The state feature vector is expected to have the form
    ``[1, x1, ..., xd]``, with a leading intercept. The returned vector keeps
    one shared intercept and has a separate block of the remaining state
    features for each action. Only the intercept and the weights for ``a`` are
    selected; all other elements are set to zero.

    Args:
        s: State to represent.
        a: Integer ID of the action to represent.
        env: Environment providing the discrete action space.
        state_features: Callable that returns ``[1, x1, ..., xd]`` for
            ``(s, env)``.

    Returns:
        Block-coded NumPy feature vector for the state-action pair ``(s, a)``.
    """
    def active_weights(a, sf_len):
        return [0] + list(range(a*sf_len+1, a*sf_len+sf_len+1))

    s = state_features(s, env)
    x = np.zeros(1+len(s)*env.action_space.n)
    x[active_weights(a, len(s)-1)] = s
    return x

def v_hat(s, w, env, state_features):
    """Compute the linear state-value approximation ``v_hat(s, w)``.

    Approximates the state value as ``w^T x(s)``, the weighted sum of the components in the
    state feature vector ``x(s)``. The weight and feature vectors must have the
    same length.

    Args:
        s: State to evaluate.
        w: Weight vector of the linear approximator.
        env: Environment containing the state.
        state_features: Callable returning the feature vector ``x(s)`` for
            ``(s, env)``.

    Returns:
        Scalar estimate of the expected return from ``s``.
    """
    return np.dot(w, state_features(s, env))

def q_hat(s, a, w, env, state_features):
    """Compute the linear action-value approximation ``q_hat(s, a, w)``.

    Estimates the q-value as ``w^T x(s, a)`` using the block-coded state-action
    features produced by `state_action_features()`. The weight vector must
    have the same length as that feature vector.

    Args:
        s: State to evaluate.
        a: Integer ID of the action to evaluate.
        w: Weight vector of the linear approximator.
        env: Environment providing the discrete action space.
        state_features: Callable returning a state feature vector with a
            leading intercept for ``(s, env)``.

    Returns:
        Scalar estimate of the expected return from taking ``a`` in ``s``.
    """    
    x = state_action_features(s, a, env, state_features)
    return np.dot(w, x)

def epsilon_greedy_action_w(
    s, w, env, state_features, epsilon=0, rng=None
):
    """Select an epsilon-greedy action from approximate action values.

    Args:
        s: Current state.
        w: Weight vector for the action-value approximator.
        env: Environment providing the discrete action space.
        state_features: Callable converting ``(state, env)`` to a feature vector.
        epsilon: Probability of selecting a uniformly random action.
        rng: NumPy generator or integer seed.

    Returns:
        Selected integer action ID.
    """
    
    rng = get_rng(rng)

    if epsilon > 0 and rng.random() < epsilon:
        return rng.integers(env.action_space.n)
    
    return random_argmax(
        [
            q_hat(s, a, w, env, state_features)
            for a in range(env.action_space.n)
        ],
        rng=rng,
    )


def MSVE(V, V_true, weight=None):
    """Calculate the weighted mean squared value error.

    Args:
        V: Estimated value for each state.
        V_true: Reference value for each state.
        weight: Weight for each state, typically its stationary visitation
            probability. If omitted, use unit weights.

    Returns:
        Weighted sum of squared value errors.
    """
    if weight is None:
        weight = np.ones(len(V))
    
    return np.sum(weight * (V - V_true)**2)


def semi_gradient_TD0_estimation(
    env,
    state_features,
    policy,
    n,
    alpha,
    gamma,
    max_episode_length=1000,
    verbose=False,
):
    """Estimate state values with semi-gradient TD(0).

    This function runs TD(0) learning with function approximation over multiple
    episodes generated from a given policy and environment. Updates are performed
    using the semi-gradient of the value function approximation.

    Args:
        env: Episodic Gymnasium environment used to generate experience.
        state_features: Callable converting ``(state, env)`` to a feature vector.
        policy: Deterministic policy indexed by state.
        n: Number of training episodes.
        alpha: Step size or schedule.
        gamma: Discount factor in ``[0, 1]``.
        max_episode_length: Maximum number of steps per episode.
        verbose: Whether to print step-by-step diagnostics.

    Returns:
        Learned weight vector for the approximate value function.
    """
    assert gamma >= 0 and gamma <= 1, "Gamma must be in [0,1]"
    assert n > 0, "Number of episodes must be positive"
    assert max_episode_length > 0, "Max episode length must be positive"
    
    if isinstance(env.observation_space, gym.spaces.Discrete):
        warnings.warn("The environment has a discrete state space. Consider using a tabular method instead of function approximation.")
    
    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)

    state, _ = env.reset()
    w = np.zeros(len(state_features(state, env)))  # Initialize weights (intercept + x and y)

    for episode in tqdm(range(n), desc="Semi-Gradient TD(0)", disable=verbose):
        state, _ = env.reset()
        done = False

        i = 0
        while not done and i < max_episode_length:
            action = policy[state]  # follow policy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Semi-gradient TD(0) update
            # Note: v_hat(terminal, w) needs to be 0
            if terminated:
                w += alpha(episode) * (reward - v_hat(state, w, env, state_features)) * state_features(state, env)
            else: 
                w += alpha(episode) * (reward + gamma * v_hat(next_state, w, env, state_features) - v_hat(state, w, env, state_features)) * state_features(state, env)
             
            if verbose:
                print (f"Episode {episode+1}, Step {i+1}: S={state}, A={action}, R={reward}, S'={next_state}, w={w}")

            state = next_state
            i += 1

    return w


def semi_gradient_Sarsa_0(env, state_features, n, epsilon, alpha, gamma, w=None,
                          max_episode_length=1000, verbose=False,
                          history=False, rng=None):
    """Run semi-gradient Sarsa(0) with function approximation.

    Implements the **semi-gradient Sarsa(0)** algorithm for estimating the optimal
    action-value function q_*(s, a) using a differentiable function approximator
    q̂(s, a, w). Actions are selected according to an ε-greedy policy derived
    from the current action-value estimate.

    Episodes are truncated after `max_episode_length` time steps.

    Args:
        env: Episodic environment used to generate experience.
        state_features: Callable converting ``(state, env)`` to a feature vector.
        n: Number of training episodes.
        epsilon: Exploration rate or schedule for the epsilon-greedy policy.
        alpha: Step size or schedule.
        gamma: Discount factor in ``[0, 1]``.
        w: Initial action-value weight vector. If omitted, initialize it to zeros.
        max_episode_length: Maximum number of steps per episode.
        verbose: Whether to print step-by-step diagnostics.
        history: Whether to return weights, returns, and episode lengths collected
            during training.
        rng: NumPy generator or integer seed for exploration and tie-breaking.

    Returns:
        Learned weight vector. If ``history`` is true, returns ``(w, history)``.
    """
    
    assert gamma >= 0 and gamma <= 1, "Gamma must be in [0,1]"
    assert n > 0, "Number of episodes must be positive"
    assert max_episode_length > 0, "Max episode length must be positive"

    rng = get_rng(rng)

    if isinstance(env.observation_space, gym.spaces.Discrete):
        warnings.warn("The environment has a discrete state space. Consider using a tabular method instead of function approximation.")

    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)
    if not isinstance(epsilon, Schedule):
        epsilon = ConstantSchedule(epsilon)

    if w is None:
        state, _ = env.reset()
        w = np.zeros(len(state_action_features(state, 0, env, state_features)))

    if history:
        ws = []
        ws.append(w.copy())
        returns = []
        ep_lens = []
    
    for episode in tqdm(range(n), desc="Semi-Gradient SARSA(0)", disable=verbose):
        state, _ = env.reset()
        action = epsilon_greedy_action_w(
            state, w, env, state_features, epsilon(episode), rng=rng
        )
        done = False

        i = 0
        if history:
            G = 0
        while not done and i < max_episode_length:
            

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            x = state_action_features(state, action, env, state_features)
            
            if terminated:
                next_action = None
                w += alpha(episode) * (reward - q_hat(state, action, w, env, state_features)) * x
                
            else:
                next_action = epsilon_greedy_action_w(
                    next_state, w, env, state_features, epsilon(episode), rng=rng
                )
                w += alpha(episode) * (reward + gamma * q_hat(next_state, next_action, w, env, state_features) - q_hat(state, action, w, env, state_features)) * x

            if verbose:
                print (f"Episode {episode+1}, Step {i+1}: S={state}, A={action}, R={reward}, S'={next_state}, w={w}")

            state = next_state
            action = next_action
            i += 1
            
            if history:
                G += reward * (gamma ** (i-1))
            
        if history:
            ws.append(w.copy())
            returns.append(G)
            ep_lens.append(i)

    if history:
        return w, {'ws': ws, 'returns': returns, 'ep_lens': ep_lens}
    
    return w  


# product from itertools is the cartesian product
def create_fourier_basis_coefs(dim, order): 
    """Create Fourier basis coefficient vectors.

    Args:
        dim: Number of state-feature dimensions.
        order: Maximum coefficient in each dimension.

    Returns:
        Array containing the Cartesian product of coefficients from zero through
        ``order`` in each dimension.
    """
    return np.array(list(product(range(order+1), repeat=dim)))
    
def transformation_fourier_basis(min, max, order):
    """Create a Fourier basis feature transformation.

    Args:
        min: Minimum state value in each dimension.
        max: Maximum state value in each dimension.
        order: Maximum Fourier coefficient in each dimension.

    Returns:
        Callable that normalizes a state to the unit hypercube and returns its
        Fourier basis features.

    Example:
        ```python
        transform = transformation_fourier_basis([0, 0], [1, 1], order=3)

        def state_features(state, env):
            return transform(state)
        ```
    """
    
    min = np.array(min)
    max = np.array(max)
    coefs = create_fourier_basis_coefs(len(min), order)
    
    def fourier_basis(s):
        # normalize state to [0,1]
        s = np.array(s)
        assert s.shape == min.shape, "State dimension does not match Fourier basis dimension"
        
        norm_s = (s - min) / (max - min)
        return np.cos(np.pi * np.dot(coefs, norm_s))
    
    return fourier_basis
