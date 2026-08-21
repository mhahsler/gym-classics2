"""This file implements temporal difference learning algorithms for policy evaluation and control in gym-classics    
environments with discrete state spaces. The algorithms include Sarsa(0) and Q-learning, which are fundamental 
methods in reinforcement learning for learning value functions and optimal policies from experience without requiring a 
model of the environment.
"""

import numpy as np
from tqdm import tqdm

import gymnasium as gym

from gym_classics2.utils import get_rng, random_argmax
from gym_classics2.algorithms.schedules import Schedule, ConstantSchedule

def Sarsa_0(env, discount, alpha, epsilon, Q=None, n=100, verbose=False,
            history=False, rng=None):
    """Learn action values with one-step on-policy Sarsa.

    Args:
        env: Gymnasium environment with discrete observation and action spaces.
        discount: Reward discount factor in ``[0, 1]``.
        alpha: Scalar step size or a
            :class:`~gym_classics2.algorithms.schedules.Schedule` evaluated once
            per episode.
        epsilon: Scalar exploration probability or a schedule evaluated once per
            episode.
        Q: Optional initial action-value array shaped
            ``(env.observation_space.n, env.action_space.n)``. The array is updated
            in place.
        n: Number of training episodes.
        verbose: Print individual updates when true.
        history: Retain Q arrays, discounted episode returns, and episode lengths.
        rng: NumPy generator or integer seed for exploration and tie-breaking.

    Returns:
        The learned Q array. If ``history=True``, returns ``(Q, history_dict)``;
        the dictionary contains ``Qs``, ``returns``, and ``ep_lens``.

    Raises:
        AssertionError: If the observation space is not discrete.
    """
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Tabular methods require discrete state space."  

    rng = get_rng(rng)
        
    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)
    if not isinstance(epsilon, Schedule):
        epsilon = ConstantSchedule(epsilon)

    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    if history:
        Q_list = []
        Q_list.append(Q.copy())
        return_list = []
        ep_len_list = []
        
    
    for i in tqdm(range(n), desc="Sarsa", disable=verbose):
        s, r = env.reset()
         
        if verbose:
            print(f"--- Episode {i} ---")      
          
        if rng.random() > epsilon(i):
            a = random_argmax(Q[s, :], rng=rng)
        else:
            a = rng.integers(env.action_space.n)
          
        t = 0
        done = False
        G = 0
        while not done:            
            sp, r, done, _, _ = env.step(a)
            G += r * pow(discount, t)
            t += 1
            
        
            if rng.random() > epsilon(i):
                ap = random_argmax(Q[sp, :], rng=rng)
            else:
                ap = rng.integers(env.action_space.n)
        
            Q[s,a] = Q[s,a] + alpha(i) * (r + discount * Q[sp,ap] - Q[s,a])
            
            if verbose:
                print(f"{t} - sarsa: {s},{a},{r},{sp},{ap}, - new Q(s,a): {Q[s,a]}")
                if done:
                    print("Total return:", G)
            
            s = sp
            a = ap
        
        if history:
            Q_list.append(Q.copy())
            return_list.append(G)   
            ep_len_list.append(t)
    
    if history:
        return Q, {'Qs': Q_list, 'returns': return_list, 'ep_lens': ep_len_list}
          
    return Q


def Q_learning(env, discount, alpha, epsilon, Q=None, n=100, verbose=False,
               history=False, rng=None):
    """Learn action values with one-step off-policy Q-learning.

    Args:
        env: Gymnasium environment with discrete observation and action spaces.
        discount: Reward discount factor in ``[0, 1]``.
        alpha: Scalar step size or a
            :class:`~gym_classics2.algorithms.schedules.Schedule` evaluated once
            per episode.
        epsilon: Scalar exploration probability or a schedule evaluated once per
            episode.
        Q: Optional initial action-value array shaped
            ``(env.observation_space.n, env.action_space.n)``. The array is updated
            in place.
        n: Number of training episodes.
        verbose: Print progress information when true.
        history: Retain Q arrays, discounted episode returns, episode lengths, and
            state-visit counts.
        rng: NumPy generator or integer seed for exploration and tie-breaking.

    Returns:
        The learned Q array. If ``history=True``, returns ``(Q, history_dict)``;
        the dictionary contains ``Qs``, ``returns``, ``ep_lens``, and
        ``state_visits``.

    Raises:
        AssertionError: If the observation space is not discrete.
    """
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Tabular methods require discrete state space."  

    rng = get_rng(rng)
    
    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)
    if not isinstance(epsilon, Schedule):
        epsilon = ConstantSchedule(epsilon)

    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    if history:
        Q_list = []
        Q_list.append(Q.copy())
        return_list = []
        ep_len_list = []
        #state_visits = np.zeros(env.observation_space.n, dtype=int)
        # Note we use float so visualization works better
        state_visits = np.zeros(env.observation_space.n, dtype=float)

    for i in tqdm(range(n), desc="Q-Learning", disable=verbose):
        s, r = env.reset()
        
        if history:
            state_visits[s] += 1
          
        done = False
        G = 0
        t = 0   
        while not done:
            # epsilon-greedy choice w.r.t. Q 
            if rng.random() > epsilon(i):
                a = random_argmax(Q[s, :], rng=rng)
            else:
                a = rng.integers(env.action_space.n)
            
            sp, r, done, _, _ = env.step(a)
        
            if history:
                state_visits[sp] += 1
        
            if history:
                G += r * pow(discount, t)
                t += 1
        
            Q[s,a] = Q[s,a] + alpha(i) * (r + discount * np.max(Q[sp,:]) - Q[s,a])
            
            s = sp
        
        if history:
            Q_list.append(Q.copy())
            return_list.append(G)
            ep_len_list.append(t)
    
    if history:
        return Q, {'Qs': Q_list, 'returns': return_list, 'ep_lens': ep_len_list, 'state_visits': state_visits}
          
    return Q
