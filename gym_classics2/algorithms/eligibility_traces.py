"""Semi-gradient Sarsa(lambda) with linear function approximation.

Callers provide a ``state_features(state, env)`` function that converts states
to feature vectors.
"""

import numpy as np
from tqdm import tqdm

from gym_classics2.algorithms.linear_approximation import (
    epsilon_greedy_action_w,
    q_hat,
    state_action_features,
)
from gym_classics2.algorithms.schedules import Schedule, ConstantSchedule
from gym_classics2.utils import get_rng

def semi_gradient_Sarsa_lambda(
    env,
    state_features,
    n,
    epsilon,
    alpha,
    gamma,
    lam,
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
        gamma: Discount factor in ``[0, 1]``.
        lam: Trace-decay parameter lambda in ``[0, 1]``.
        w: Initial weight vector. If omitted, initialize it to zeros.
        max_episode_length: Maximum number of steps per episode.
        verbose: Whether to print step-by-step diagnostics.
        history: Whether to return weights, returns, and episode lengths collected
            during training.
        rng: NumPy generator or integer seed for exploration and tie-breaking.

    Returns:
        Learned weight vector. If ``history`` is true, returns ``(w, history)``.
    """

    assert gamma >= 0 and gamma <= 1, "gamma must be in [0,1]"
    assert lam >= 0 and lam <= 1, "lambda must be in [0,1]"
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
            env, w, state, state_features, epsilon(episode), rng=rng
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
            
            G += reward * (gamma ** i)  # accumulate return if history is enabled

            # current feature vector for (state, action)
            x = state_action_features(state, action, env, state_features)

            # update trace
            z = gamma * lam * z + (1 - alpha(episode) * gamma * lam * np.dot(z, x)) * x

            if terminated:
                delta = reward - q_hat(state, action, w, env, state_features)
            else:
                next_action = epsilon_greedy_action_w(
                    env, w, next_state, state_features, epsilon(episode), rng=rng
                )
                delta = reward + gamma * q_hat(next_state, next_action, w, env, state_features) - q_hat(state, action, w, env, state_features)

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
