"""This file implements different policy representations and functions for working with policies in gym-classics environments.
It includes functions for creating random policies, encoding policies for display, and computing greedy policies based
"""

import numpy as np

import gymnasium as gym

from gym_classics2.utils import get_rng, random_argmax
from gym_classics2.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

def make_multidiscrete_policy(policy, env):
    """
    Converts a tabular policy vector to a multi-discrete tabular policy stored in a dictionary that can be used for sample.
    """
    assert isinstance(env.observation_space, gym.spaces.MultiDiscrete), "Requires an environment with a multi-discrete state spaces."
    
    if isinstance(policy, dict):
        return policy
        
    if isinstance(env, GymClassicsBaseEnv):
        return dict(zip(env.id2state(list(env.states())), [policy[s] for s in env.states()]))
    else:
        raise ValueError("Unsupported environment type")


def random_policy(env, rng=None):
    """
    Create a random policy for the given environment. 
    The policy is represented as a numpy array where each entry corresponds to an action for a state.
    ``rng`` may be a NumPy generator or an integer seed.
    """
    
    rng = get_rng(rng)

    if isinstance(env.observation_space, gym.spaces.Discrete):
        return rng.integers(env.action_space.n, size=env.observation_space.n)
    else:
        assert isinstance(env, GymClassicsBaseEnv), "Only gym-classics environments are supported for random policies with multi-discrete state spaces."
        return make_multidiscrete_policy(rng.integers(env.action_space.n, size=len(env.states())), env)

# only for gym-classics environments!
def encode_policy(policy, env, type = "text"):
    """
    Encode a policy for display. The policy is represented as a numpy array where each entry corresponds to an action for a state.
    The function returns a list of action names corresponding to the actions in the policy.
    """
    
    assert isinstance(env, GymClassicsBaseEnv)
    
    return [env.unwrapped.id2action(a, type = type) for a in policy]


# this is a copy from DP to prevent circular imports.
def _backup(env, discount, V, s, a):
    V = np.array(V)
    
    next_states, rewards, terminals, probs = env.model(s, a)
    bootstraps = (1.0 - terminals) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))

def greedy_policy(V, env, discount=1, rng=None):
    """Calculate a greedy policy from a state-value function.

    Args:
        V: One-dimensional state-value array.
        env: Environment with a discrete state space and model access.
        discount: Discount factor in ``[0, 1]``.
        rng: NumPy generator or integer seed for random tie-breaking.

    Returns:
        Integer action ID selected for each state.
    """
    assert isinstance(env, GymClassicsBaseEnv), "greedy_policy requires a gym-classics environment with discrete state space."
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert 0.0 <= discount <= 1.0
    
    policy = np.zeros(len(env.states()), dtype=np.int64)
    rng = get_rng(rng)

    env = env.unwrapped
    
    for s in env.states():
        Q_values = [_backup(env, discount, V, s, a) for a in range(env.action_space.n)]
        policy[s] = random_argmax(Q_values, rng=rng)

    return policy

def greedy_policy_Q(Q, env, discount=1, rng=None):
    """Calculate a greedy policy from an action-value function.

    Args:
        Q: Two-dimensional action-value array indexed by state and action.
        env: Environment with discrete observation and action spaces.
        discount: Unused; retained for API compatibility with ``greedy_policy``.
        rng: NumPy generator or integer seed for random tie-breaking.

    Returns:
        Integer action ID selected for each state.
    """
    
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert 0.0 <= discount <= 1.0

    return random_argmax(Q, axis=1, rng=rng)

def epsilon_greedy_action(policy, state=None, epsilon=0, rng=None):
    """Select an epsilon-greedy action from tabular action values.

    Args:
        policy: Action values for all actions, optionally indexed first by state.
        state: Current state index. If omitted, ``policy`` is treated as the
            action-value vector for the current state.
        epsilon: Probability of selecting a uniformly random action.
        rng: NumPy generator or integer seed.

    Returns:
        Selected integer action ID.
    """
    
    rng = get_rng(rng)

    if epsilon > 0 and rng.random() < epsilon:
        return rng.integers(len(policy)) if state is None else rng.integers(len(policy[state]))
    
    if state is None:
        return random_argmax(policy, rng=rng)
    else:
        return random_argmax(policy[state], rng=rng)
