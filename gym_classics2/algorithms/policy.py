"""Create, convert, and select actions from tabular policies.

A deterministic tabular policy is a vector of integer action IDs indexed by
state ID. For gym-classics2 environments with MultiDiscrete observations,
policies can instead be dictionaries mapping coordinate tuples to action IDs.

Greedy policies can be derived from state values using model access or directly
from action values. ``epsilon_greedy_action`` samples from action values rather
than from a deterministic policy vector. Random choices accept a NumPy
generator, an integer seed, or None through ``rng``.
"""

import numpy as np

import gymnasium as gym

from gym_classics2.utils import get_rng, random_argmax
from gym_classics2.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv

def random_policy(env, rng=None):
    """Create a deterministic policy by sampling one action per state.

    Actions are drawn uniformly and independently when the policy is created.
    Looking up a state in the resulting policy always gives the same action.

    Args:
        env: Environment with a discrete action space. Observations must be
            Discrete, or MultiDiscrete in a gym-classics2 environment.
        rng: NumPy generator, integer seed, or None for sampling actions.

    Returns:
        NumPy array of action IDs indexed by state ID for Discrete observations,
        or a dictionary mapping coordinate tuples to action IDs for
        MultiDiscrete observations.
    """
    
    rng = get_rng(rng)

    if isinstance(env.observation_space, gym.spaces.Discrete):
        return rng.integers(env.action_space.n, size=env.observation_space.n)
    else:
        assert isinstance(env, GymClassicsBaseEnv), "Only gym-classics environments are supported for random policies with multi-discrete state spaces."
        return make_multidiscrete_policy(rng.integers(env.action_space.n, size=len(env.states())), env)

# only for gym-classics environments!
def encode_policy(policy, env, type = "text"):
    """Convert a sequence of action IDs to labels for display.

    Args:
        policy: Deterministic policy vector containing integer action IDs.
        env: gym-classics2 environment providing ``id2action``.
        type: Label format passed to ``env.id2action``; defaults to ``"text"``.
            Available formats depend on the environment.

    Returns:
        List of action labels in the same order as the input policy.
    """
    
    assert isinstance(env, GymClassicsBaseEnv)
    
    return [env.unwrapped.id2action(a, type = type) for a in policy]


# this is a copy from DP to prevent circular imports.
def _backup(env, discount, V, s, a):
    """Return the expected one-step target for state ``s`` and action ``a``.

    Uses ``env.model(s, a)`` to sum ``p * (reward + discount * V[next_state])``
    over outcomes, omitting the next-state value for terminal transitions.
    ``V`` is indexed by state ID.
    """
    V = np.array(V)
    
    next_states, rewards, terminals, probs = env.model(s, a)
    bootstraps = (1.0 - terminals) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))

def greedy_policy(V, env, discount=1, rng=None):
    """Create a greedy policy by one-step look-ahead from state values.

    For each state, select an action maximizing the expected reward plus the
    discounted next-state value. Terminal transitions do not bootstrap.
    Ties are broken uniformly at random when the policy is created.

    Args:
        V: One-dimensional state-value array indexed by state ID.
        env: gym-classics2 environment with Discrete observation and action
            spaces and access to the transition and reward model.
        discount: Discount factor in ``[0, 1]``.
        rng: NumPy generator, integer seed, or None for random tie-breaking.

    Returns:
        NumPy array of integer action IDs indexed by state ID.
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
    """Create a greedy policy by maximizing action values in each state.

    Ties are broken uniformly at random when the policy is created.

    Args:
        Q: Two-dimensional action-value array indexed by state and action.
        env: Environment with discrete observation and action spaces.
        discount: Must be in ``[0, 1]`` but does not affect action selection;
            retained for API compatibility with ``greedy_policy``.
        rng: NumPy generator, integer seed, or None for random tie-breaking.

    Returns:
        NumPy array of integer action IDs indexed by state ID.
    """
    
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    assert 0.0 <= discount <= 1.0

    return random_argmax(Q, axis=1, rng=rng)

def epsilon_greedy_action(policy, state=None, epsilon=0, rng=None):
    """Select an epsilon-greedy action from tabular action values.

    With probability ``epsilon``, sample uniformly from all actions, including
    greedy ones. Otherwise, maximize the action values and break ties
    uniformly at random.

    Args:
        policy: One-dimensional action-value vector, or a table indexed by
            state and then action. Despite the name, this argument contains
            action values, not the action IDs of a deterministic policy.
        state: Current state index. If omitted, ``policy`` is treated as the
            action-value vector for the current state.
        epsilon: Exploration probability in ``[0, 1]``.
        rng: NumPy generator, integer seed, or None for action sampling
            and tie-breaking.

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

def make_multidiscrete_policy(policy, env):
    """Convert a policy indexed by state ID to a coordinate-keyed dictionary.

    Args:
        policy: Deterministic policy vector indexed by state ID, or an existing
            dictionary mapping coordinate tuples to action IDs.
        env: Environment with MultiDiscrete observations. Converting a vector
            requires a gym-classics2 environment providing ``states`` and
            ``id2state``.

    Returns:
        Dictionary mapping coordinate tuples to integer action IDs. An input
        dictionary is returned unchanged, without copying or validating entries.

    Raises:
        AssertionError: If the observation space is not MultiDiscrete.
        ValueError: If converting a vector for an unsupported environment type.
    """
    assert isinstance(env.observation_space, gym.spaces.MultiDiscrete), "Requires an environment with a multi-discrete state spaces."
    
    if isinstance(policy, dict):
        return policy
        
    if isinstance(env, GymClassicsBaseEnv):
        return dict(zip(env.id2state(list(env.states())), [policy[s] for s in env.states()]))
    else:
        raise ValueError("Unsupported environment type")


