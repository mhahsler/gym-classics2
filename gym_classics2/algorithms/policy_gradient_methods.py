"""Policy-gradient control with linear function approximation.

REINFORCE uses sampled returns; one-step actor-critic (AC) uses a learned
state-value function to estimate the policy update. Both learn a stochastic
softmax policy over linear action preferences for a discrete action space.

Callers supply ``state_features(state, env)``, returning a fixed-length NumPy
vector with a leading intercept, such as ``[1, x, y]``. Policy features are
constructed by ``state_action_features``; AC also uses the state features
directly for its value approximation.
"""

from tqdm import tqdm
import numpy as np
import warnings

import gymnasium as gym

from gym_classics2.algorithms.linear_approximation import state_action_features
from gym_classics2.algorithms.schedules import ConstantSchedule
from gym_classics2.utils import get_rng

def h(s, a, theta, env, state_features):
    """Compute the linear action preference ``h(s, a) = theta^T x(s, a)``.

    Args:
        s: State to evaluate.
        a: Integer action ID in ``range(env.action_space.n)``.
        theta: Policy parameter vector matching ``state_action_features``.
        env: Environment providing the discrete action space.
        state_features: Callable returning state features with a leading
            intercept for ``(s, env)``.

    Returns:
        Scalar preference used as the action's softmax logit.
    """
    return np.dot(theta, state_action_features(s, a, env, state_features))

def pi(s, theta, env, state_features):
    """Compute the softmax policy ``pi(a | s)`` over linear preferences.

    Args:
        s: State to evaluate.
        theta: Policy parameter vector matching ``state_action_features``.
        env: Environment providing the discrete action space.
        state_features: Callable returning state features with a leading
            intercept for ``(s, env)``.

    Returns:
        NumPy vector of length ``env.action_space.n``, indexed by action ID,
        with probabilities proportional to ``exp(h(s, a))``.
    """
    hs = np.array(
        [h(s, a, theta, env, state_features) for a in range(env.action_space.n)]
    )
    exp_hs = np.exp(hs)
    return exp_hs / np.sum(exp_hs)

def sample_episode_approx_policy(
    env, pi, theta, state_features, max_episode_length=1000, rng=None
):
    """Sample an episode from a parameterized policy.

    Args:
        env: Episodic Gymnasium environment with a discrete action space.
        pi: Callable ``pi(state, theta, env, state_features)`` returning an
            action-probability vector indexed by action ID.
        theta: Policy parameter vector.
        state_features: Feature callable passed through to ``pi``.
        max_episode_length: Maximum number of transitions to sample.
        rng: NumPy generator, integer seed, or None for action sampling.
            Does not seed the environment.

    Returns:
        List of ``(state, action, reward, next_state)`` transitions, beginning
        with a fresh environment reset. Stops on termination or after
        ``max_episode_length`` transitions; truncation is currently ignored.
    """

    rng = get_rng(rng)
    s, _ = env.reset()
    episode_data = []
    
    for t in range(max_episode_length):
        a = rng.choice(
            env.action_space.n, p=pi(s, theta, env, state_features)
        )
        next_s, r, done, _, _ = env.step(a)
        episode_data.append((s, a, r, next_s))
        s = next_s
        
        if done:
            break
    
    return episode_data

def choose_action_w(env, pi, theta, state, state_features, rng=None):
    """Sample an action from a parameterized policy.

    Args:
        env: Environment providing the discrete action space.
        pi: Callable ``pi(state, theta, env, state_features)`` returning an
            action-probability vector indexed by action ID.
        theta: Policy parameter vector.
        state: Current state.
        state_features: Feature callable passed through to ``pi``.
        rng: NumPy generator, integer seed, or None for action sampling.

    Returns:
        Selected integer action ID.
    """
    rng = get_rng(rng)
    return rng.choice(
        env.action_space.n, p=pi(state, theta, env, state_features)
    )

def REINFORCE(
    env,
    state_features,
    n,
    alpha,
    gamma,
    theta = None,
    max_episode_length=1000,
    verbose=False,
    history=False,
    rng=None,
    ):
    """Learn a linear softmax policy with episodic REINFORCE (no baseline).

    Sample an episode, then update ``theta`` at each visited state using
    ``alpha(episode) * gamma**t * G_t * grad_log_pi``, where ``G_t`` is the
    discounted return from step ``t`` to the end of the sampled trajectory.

    Args:
        env: Episodic Gymnasium environment with a discrete action space.
        state_features: Callable returning a fixed-length NumPy feature vector
            with a leading intercept for ``(state, env)``.
        n: Positive number of training episodes.
        alpha: Float step size or callable of the zero-based episode index.
        gamma: Discount factor in ``[0, 1]``.
        theta: Initial NumPy policy parameter vector, updated in place. Its
            length must match ``state_action_features``. Defaults to zeros,
            giving a uniform initial policy.
        max_episode_length: Positive maximum number of steps per episode.
        verbose: Print step-by-step diagnostics instead of a progress bar.
        history: Whether to return training history alongside the parameters.
        rng: NumPy generator, integer seed, or None for action sampling.
            Does not seed the environment.

    Returns:
        Learned parameter vector ``theta``, or ``(theta, history)`` when
        ``history=True``. The history dictionary contains ``thetas`` (copies
        before training and after each episode), ``returns`` (discounted
        episode returns), and ``ep_lens`` (episode lengths in steps).

    Notes:
        Sampling ignores environment truncation. If the step limit cuts an
        episode short, returns use only the sampled rewards, without
        bootstrapping a value for the final state.
    """
    
    assert gamma >= 0 and gamma <= 1, "gamma must be in [0  ,1]"
    assert n > 0, "number of episodes must be positive"
    assert max_episode_length > 0, "max episode length must be positive"

    rng = get_rng(rng)
    
    if isinstance(env.observation_space, gym.spaces.Discrete):
        warnings.warn("The environment has a discrete state space. Consider using a tabular method instead of function approximation.")
    
    if isinstance(alpha, float):
        alpha = ConstantSchedule(alpha)

    if theta is None:
        state, _ = env.reset()
        theta = np.zeros(
            len(state_action_features(state, 0, env, state_features))
        )
    
    if history:
        returns = []        
        ep_lens = []
        thetas = []
        thetas.append(theta.copy())
        
    for episode in tqdm(range(n), desc="Episodes", disable=verbose):
        if verbose:
            print(f"Episode {episode+1}/{n}")
        
        # sample complete episode using pi (this is a MC method)
        episode_data = sample_episode_approx_policy(
            env, pi, theta, state_features, max_episode_length, rng=rng
        )
        
        for t in range(len(episode_data)):
            # update policy for each step in the episode using the return observed from that step on
            #print(episode_data[t])
        
            G = np.sum([e[2] for e in episode_data[t:]] * (gamma ** np.arange(len(episode_data[t:]))))
            if history and t == 0:
                returns.append(G)   

            s,a,r,next_s = episode_data[t]
            
            # ln policy gradient= x(s,a)- sum_b pi(b|s,theta) x(s,b)
            grad_log_pi = state_action_features(s, a, env, state_features) - sum(
                pi(s, theta, env, state_features)[b]
                * state_action_features(s, b, env, state_features)
                for b in range(env.action_space.n)
            )
            
            if verbose: 
                print (f"t: {t}, G: {G:.2f}, grad_log_pi: {grad_log_pi}")
            
            theta += alpha(episode) * (gamma**t) * G * grad_log_pi
            
        if history:
            thetas.append(theta.copy())
            ep_lens.append(len(episode_data))
            
    if history:
        return theta, {'thetas': thetas, 'returns': returns, 'ep_lens': ep_lens}
       
    return theta


def AC(
    env,
    state_features,
    n,
    alpha_policy,
    alpha_value,
    gamma,
    max_episode_length=1000,
    verbose=False,
    history=False,
    rng=None,
    ):
    """Learn a linear softmax policy with one-step actor-critic.

    The critic approximates ``v(s) = w^T x(s)`` and learns from the TD error
    ``delta = r + gamma * v(next_state) - v(state)``. The actor updates its
    policy using ``alpha_policy(episode) * gamma**t * delta * grad_log_pi``.
    Both parameter vectors start at zero, giving a uniform initial policy.

    Args:
        env: Episodic Gymnasium environment with a discrete action space.
        state_features: Callable returning a fixed-length NumPy feature vector
            with a leading intercept for ``(state, env)``. Used by the critic
            directly and by the actor through ``state_action_features``.
        n: Positive number of training episodes.
        alpha_policy: Float policy step size or callable of the zero-based
            episode index.
        alpha_value: Float value-function step size or callable of the
            zero-based episode index.
        gamma: Discount factor in ``[0, 1]``.
        max_episode_length: Positive maximum number of steps per episode.
        verbose: Print episode numbers instead of a progress bar.
        history: Whether to return training history alongside the parameters.
        rng: NumPy generator, integer seed, or None for action sampling.
            Does not seed the environment.

    Returns:
        Tuple ``(theta, w)`` containing the learned policy and value parameter
        vectors, or ``(theta, w, history)`` when ``history=True``. The history
        dictionary contains ``thetas`` (copies before training and after each
        episode), ``returns`` (discounted episode returns), and ``ep_lens``
        (episode lengths in steps).

    Notes:
        This implementation ignores environment truncation and bootstraps
        from the next state's value even on a terminal transition. For an
        episodic task, terminal-state features should therefore yield a
        value of zero.
    """
    
    assert gamma >= 0 and gamma <= 1, "gamma must be in [0  ,1]"
    assert n > 0, "number of episodes must be positive"
    assert max_episode_length > 0, "max episode length must be positive"

    rng = get_rng(rng)
    
    if isinstance(env.observation_space, gym.spaces.Discrete):
        warnings.warn("The environment has a discrete state space. Consider using a tabular method instead of function approximation.")
    
    if isinstance(alpha_policy, float):
        alpha_policy = ConstantSchedule(alpha_policy)
    if isinstance(alpha_value, float):
        alpha_value = ConstantSchedule(alpha_value) 
    
    state, _ = env.reset()
    
    # for simplicity we use the same features for the value function and the policy approximation
    # value function weights
    w = np.zeros(len(state_features(state, env)))
    # policy weights
    theta = np.zeros(len(state_action_features(state, 0, env, state_features)))
    
    if history:
        returns = []        
        ep_lens = []
        ws = []
        ws.append(w.copy())
        thetas = []
        thetas.append(theta.copy())
        
    for episode in tqdm(range(n), desc="Episodes", disable=verbose):
        if verbose:
            print(f"Episode {episode+1}/{n}")
        
        disc_factor = 1.0
        state, _ = env.reset()
        done = False
        i = 0
        G = 0.0
        
        while not done and  i < max_episode_length:
            # use actor to determine next action
            a = rng.choice(
                env.action_space.n,
                p=pi(state, theta, env, state_features),
            )

            # execute action
            next_state, r, done, _, _ = env.step(a)
            
            # use critic to calculate  TD error
            td_error = r + gamma * np.dot(w, state_features(next_state, env)) - np.dot(w, state_features(state, env))
            
            # update critic
            w += alpha_value(episode) * td_error * state_features(state, env)
            
            # update actor
            grad_log_pi = state_action_features(
                state, a, env, state_features
            ) - sum(
                pi(state, theta, env, state_features)[b]
                * state_action_features(state, b, env, state_features)
                for b in range(env.action_space.n)
            )
            theta += alpha_policy(episode) * disc_factor * td_error * grad_log_pi
        
            G += disc_factor * r
            disc_factor *= gamma      
            state = next_state
            i += 1
            
        if history:
            thetas.append(theta.copy())
            ws.append(w.copy())
            returns.append(G)
            ep_lens.append(i)
            
    if history:
        return theta, w, {'thetas': thetas, 'returns': returns, 'ep_lens': ep_lens}
       
    return theta, w
