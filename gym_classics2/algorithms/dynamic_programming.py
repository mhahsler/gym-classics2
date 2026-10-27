"""This file implements dynamic programming algorithms for solving Markov Decision Processes (MDPs) 
in tabular gym-classics2 environments with model access.
The algorithms include value iteration and policy iteration, which are fundamental methods in reinforcement learning for 
computing optimal policies and value functions.
"""

import numpy as np
from tqdm import tqdm

from gym_classics2.utils import random_argmax
from gym_classics2.algorithms.policy import random_policy
from gym_classics2.envs.abstract.base_env import BaseEnv as GymClassicsBaseEnv


def backup(env, discount, V, state, action):
    """Computes a Bellman backup for a given state, action, and value function. This is the building block
    for the other algorithms.
    
    Args:
        env: A gym-classics environment with model access.
        discount: The discount factor.
        V: The current value function.
        state: The current state.
        action: The action to evaluate.
    Returns:
        The computed Q-value for the given state and action.
    """

    V = np.array(V)

    next_states, rewards, terminals, probs = env.model(state, action)
    bootstraps = (1.0 - terminals) * V[next_states]
    return np.sum(probs * (rewards + discount * bootstraps))


def policy_evaluation(env, discount, policy, precision=1e-3, max_backups=1000, verbose=False):
    """Performs (modified) policy evaluation to approximate the given policy's value function.
    Regular policy evaluation does not limit the number of backups and converges to the true value funciton. 
    Modified policy evaluation limits the number of backups and produces an estimate of the true value function. 
    This has several advantages: It is faster and it
    prevents infinite loops for undiscounted problems where the evaluated policy does not reach a terminal state. 
    
    Policy iteration using modified policy evaluation still converges. Note: Value iteration is a special case of 
    policy iteration where policy evaluation is limited to a single backup.

    Args:
        env: A gym-classics environment with model access.
        discount: The discount factor (0 <= discount <= 1).
        policy: The policy to evaluate.
        precision: The precision for convergence (default: 1e-3).
        max_backups: Maximum number of sweeps over all states (default: 1000).
            Limits computation and prevents infinite loops for undiscounted problems.
        verbose: If True, prints progress information.

    Returns:
        The value function for the given policy.
    """
    
    assert isinstance(env, GymClassicsBaseEnv), "Value iteration requires a gym-classics environment with model access to the environment." 
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    
    V = np.zeros(len(policy), dtype=np.float64)

    i = 0
    while i < max_backups:
        V_old = V.copy()

        for s in env.states():
            V[s] = backup(env, discount, V, s, policy[s])

        if np.abs(V - V_old).max() <= precision:
            break

        i += 1
    
    if verbose:
        if i < max_backups:
            print(f'\nConverged after {i} sweeps.')
        else:       
            print(f'\nReached maximum number of backups ({max_backups}) without convergence.')
    
    return V


def value_iteration(env, discount, precision=1e-3, history = False, verbose = False):
    """Performs value iteration to calculate the optimal value function.

    Args:
        env: A gym-classics environment with model access.
        discount: The discount factor (0 <= discount <= 1).
        precision: The precision for convergence (default: 1e-3).
        history: If True, returns a list of intermediate value functions.
        verbose: If True, prints progress information.

    Returns:
        The optimal value function V. If history is True, returns a list of intermediate value functions.
    """
    
    assert isinstance(env, GymClassicsBaseEnv), "Value iteration requires a gym-classics environment with model access to the environment." 
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0
    
    V = np.zeros(len(env.states()), dtype=np.float64)  
    if history:
        V_list = []
        V_list.append(V.copy())

    sweeps = 0
    progress = tqdm(total=None, desc="Value Iteration", disable=verbose)
    while True:
        progress.update()
        if verbose:
            print('.', end = '')
            sweeps += 1
        
        V_old = V.copy()

        for s in env.states():
            Q_values = [backup(env, discount, V, s, a) for a in range(len(env.actions()))]
            V[s] = np.max(Q_values)

        if history:
            V_list.append(V.copy())

        if np.abs(V - V_old).max() <= precision:
            break

    if verbose:
        print(f'\nConverged after {sweeps} sweeps.')

    if history:
        return V_list 
    
    return V


def policy_improvement(policy, V_policy, env, discount, precision=1e-3):
    """Improves the policy based on the given value function. It does this by 
    calculating a Q-value function for all states and actions using Bellman updates 
    and then extracts a greedy policy.

    Policy evaluation adn policy improvement are the steps of policy iteration.
    
    Args:
        policy: The current policy to improve.
        V_policy: The value function of the current policy.
        env: A gym-classics environment with model access.
        discount: The discount factor (0 <= discount <= 1).
        precision: The precision for determining stability (default: 1e-3).
        
    Returns:
        A tuple (improved_policy, stable) where stable is True if the policy did not change.
    """
    
    policy_old = policy.copy()
    V_old = V_policy.copy()

    for s in env.states():
        Q_values = [backup(env, discount, V_policy, s, a) for a in env.actions()]
        policy[s] = np.argmax(Q_values)
        V_policy[s] = max(Q_values)

    stable = np.logical_or(
        policy == policy_old,
        np.abs(V_policy - V_old).max() <= precision,
    ).all()

    return policy, stable

def policy_iteration(env, discount, precision=1e-3, max_backups=1000, history=False, verbose=False, rng=None):
    """Performs policy iteration to find the optimal policy. It repeats modified policy evaluation followed by
    policy improvement till it converges to the optimal policy.

    Using modified policy evaluation has several advantages: It is faster and it
    prevents infinite loops for undiscounted problems where the evaluated policy does not reach a terminal state.
    Policy iteration using modified policy evaluation still converges. 
    Note: Value iteration is a special case of policy iteration where policy evaluation is limited to a 
    single backup.  

    Args:
        env: A gym-classics environment with model access.
        discount: The discount factor (0 <= discount <= 1).
        precision: The precision for convergence (default: 1e-3).
        max_backups: Maximum number of iterations used in modified policy evaluation. 
        history: If True, returns lists of intermediate policies and value functions.
        verbose: If True, prints progress information.
        rng: NumPy generator or integer seed used to initialize the policy.

    Returns:
        The optimal policy. If history is True, returns a tuple (policy_list, V_list) containing lists of intermediate policies and value functions.
    """
    
    assert isinstance(env, GymClassicsBaseEnv), "Value iteration requires a gym-classics environment with model access to the environment." 
    assert 0.0 <= discount <= 1.0
    assert precision > 0.0

    policy = random_policy(env, rng=rng)

    if history:
        pol_list = []
        pol_list.append(policy.copy())
        V_list = []

    iterations = 0
    progress = tqdm(total=None, desc="Policy Iteration", disable=verbose)
    while True:
        progress.update()
        if verbose:
            print('.', end = '')
            iterations += 1

        V_policy = policy_evaluation(env, discount, policy, precision, max_backups)
        if history:
            V_list.append(V_policy.copy())

        policy, stable = policy_improvement(policy, V_policy, env, discount, precision)
             
        if stable:
            break

        if history:
            pol_list.append(policy.copy())

    if verbose:
        print(f'\nConverged after {iterations} iterations.')
        
    if history:
        return pol_list, V_list

    return policy
