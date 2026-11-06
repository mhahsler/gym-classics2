"""Sarsa(lambda) control with eligibility traces.

``Sarsa_lambda`` is the tabular version for discrete environments.
``semi_gradient_Sarsa_lambda`` uses linear function approximation.

Callers provide a ``state_features(state, env)`` function that converts states
to feature vectors.

Approximation calculations raise ``FloatingPointError`` on NumPy overflow or
invalid operations. Check the step size and feature values if this occurs.
"""

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from gym_classics2.algorithms.linear_approximation import (
    epsilon_greedy_action_w,
    q_hat,
    state_action_features,
)
from gym_classics2.algorithms.schedules import Schedule, ConstantSchedule
from gym_classics2.utils import get_rng, random_argmax

def Sarsa_lambda(env, discount, alpha, epsilon, lambda_, Q=None, n=100, verbose=False,
                 history=False, rng=None):
    """Learn action values with tabular on-policy Sarsa(lambda).

    After each step, every state-action pair is updated in proportion to its
    eligibility trace, ``Q += alpha * delta * z``, and the traces then decay by
    ``discount * lambda_``. The trace of the visited pair ``(s, a)`` can be updated
    in three ways:

    - accumulating: ``z[s, a] += 1``. Revisits add up, so traces can exceed 1
      and become unstable for large ``alpha`` or ``lambda_``.
    - replacing: ``z[s, a] = 1``. Traces are capped at 1, which often works
      better when states are revisited.
    - dutch: ``z[s, a] = (1 - alpha) * z[s, a] + 1``. A compromise between the
      two, used by true online Sarsa(lambda).

    This implementation uses **replacing traces**.

    Args:
        env: Gymnasium environment with discrete observation and action spaces.
        discount: Reward discount factor in ``[0, 1]``.
        alpha: Scalar step size or ``Schedule`` evaluated once per episode.
        epsilon: Scalar exploration probability or ``Schedule`` evaluated once
            per episode.
        lambda_: Trace-decay parameter in ``[0, 1]``. ``lambda_=0`` is Sarsa(0).
        Q: Optional initial action-value array shaped
            ``(env.observation_space.n, env.action_space.n)``. The array is updated
            in place.
        n: Number of training episodes.
        verbose: Print individual updates when true.
        history: Retain discounted episode returns and episode lengths.
        rng: NumPy generator or integer seed for exploration and tie-breaking.

    Returns:
        The learned Q array. If ``history=True``, returns ``(Q, history_dict)``;
        the dictionary contains ``returns`` and ``ep_lens``.

    Raises:
        AssertionError: If the observation space is not discrete or ``discount``
            or ``lambda_`` is outside ``[0, 1]``.
    """
    assert isinstance(env.observation_space, gym.spaces.Discrete), "Tabular methods require discrete state space."
    assert 0 <= discount <= 1, "discount must be in [0, 1]"
    assert 0 <= lambda_ <= 1, "lambda_ must be in [0, 1]"

    rng = get_rng(rng)

    if not isinstance(alpha, Schedule):
        alpha = ConstantSchedule(alpha)
    if not isinstance(epsilon, Schedule):
        epsilon = ConstantSchedule(epsilon)

    if Q is None:
        Q = np.zeros((env.observation_space.n, env.action_space.n))

    def choose(s, eps):
        if rng.random() < eps:
            return rng.integers(env.action_space.n)
        return random_argmax(Q[s, :], rng=rng)

    return_list, ep_len_list = [], []
    for i in tqdm(range(n), desc="Sarsa(lambda)", disable=verbose):
        s, _ = env.reset()
        a = choose(s, epsilon(i))
        z = np.zeros_like(Q)
        done, t, G = False, 0, 0

        while not done:
            sp, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            G += r * discount**t
            t += 1

            ap = choose(sp, epsilon(i))
            target = r if terminated else r + discount * Q[sp, ap]
            delta = target - Q[s, a]

            z *= discount * lambda_
            z[s, a] = 1
            Q += alpha(i) * delta * z
         

            if verbose:
                print(f"{t} - sarsa(lambda): {s},{a},{r},{sp},{ap}, delta: {delta}")
            s, a = sp, ap

        return_list.append(G)
        ep_len_list.append(t)

    if history:
        return Q, {'returns': return_list, 'ep_lens': ep_len_list}

    return Q


@np.errstate(over="raise", invalid="raise")
def semi_gradient_Sarsa_lambda(
    env,
    state_features,
    n,
    epsilon,
    alpha,
    discount,
    lambda_,
    w=None,
    max_episode_length=1000,
    verbose=False,
    history=False,
    rng=None,
):
    """Run semi-gradient Sarsa(lambda) with linear function approximation.

    Args:
        env: Episodic environment used to generate experience.
        state_features: Callable converting ``(state, env)`` to a feature vector.
        n: Number of episodes.
        epsilon: Exploration rate or schedule for the epsilon-greedy policy.
        alpha: Step size or schedule.
        discount: Discount factor in ``[0, 1]``.
        lambda_: Trace-decay parameter lambda in ``[0, 1]``.
        w: Initial weight vector. If omitted, initialize it to zeros.
        max_episode_length: Maximum number of steps per episode.
        verbose: Whether to print step-by-step diagnostics.
        history: Whether to return weights, returns, and episode lengths collected
            during training.
        rng: NumPy generator or integer seed for exploration and tie-breaking.

    Returns:
        Learned weight vector. If ``history`` is true, returns ``(w, history)``.
    """

    assert discount >= 0 and discount <= 1, "discount must be in [0,1]"
    assert lambda_ >= 0 and lambda_ <= 1, "lambda_ must be in [0,1]"
    assert n > 0, "number of episodes must be positive"
    assert max_episode_length > 0, "max episode length must be positive"

    rng = get_rng(rng)

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


    for episode in tqdm(range(n), desc="Semi-Gradient SARSA(lambda)", disable=verbose):
        state, _ = env.reset()
        action = epsilon_greedy_action_w(
            state, w, env, state_features, epsilon(episode), rng=rng
        )

        # eligibility trace vector, same size as w
        z = np.zeros_like(w)
        Q_old = 0

        done = False
        i = 0
        
        G = 0  # for tracking returns if history is enabled

        while not done and i < max_episode_length:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            G += reward * (discount ** i)  # accumulate return if history is enabled

            # current feature vector for (state, action)
            x = state_action_features(state, action, env, state_features)

            # update trace
            z = discount * lambda_ * z + (1 - alpha(episode) * discount * lambda_ * np.dot(z, x)) * x

            if terminated:
                delta = reward - q_hat(state, action, w, env, state_features)
            else:
                next_action = epsilon_greedy_action_w(
                    next_state, w, env, state_features, epsilon(episode), rng=rng
                )
                delta = reward + discount * q_hat(next_state, next_action, w, env, state_features) - q_hat(state, action, w, env, state_features)

            # semi-gradient weight update
            Q = q_hat(state, action, w, env, state_features)
            Q_prime = q_hat(next_state, next_action, w, env, state_features) if not terminated else 0
            w += alpha(episode) * (delta + Q - Q_old) * z - alpha(episode) * (Q - Q_old) * x

            Q_old = Q_prime

            if verbose:
                if terminated:
                    print(
                        f"Episode {episode+1}, Step {i+1}: "
                        f"S={state}, A={action}, R={reward}, S'={next_state}, "
                        f"delta={delta}, z={z}, w={w}"
                    )
                else:
                    print(
                        f"Episode {episode+1}, Step {i+1}: "
                        f"S={state}, A={action}, R={reward}, S'={next_state}, A'={next_action}, "
                        f"delta={delta}, z={z}, w={w}"
                    )

            if done:
                break

            state = next_state
            action = next_action
            i += 1

        if history:
            returns.append(G)
            ws.append(w.copy())
            ep_lens.append(i)
            
        
    if history:        
        return w, {'ws': ws, 'returns': returns, 'ep_lens': ep_lens}
        
    return w
