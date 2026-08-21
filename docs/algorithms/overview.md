# Algorithms overview

The included implementations are organized by the information available to the
agent and by the form of the value function.

| Algorithm | Module | Model? | State-space requirement | Main result |
| --- | --- | ---: | --- | --- |
| Value iteration | `dynamic_programming` | Yes | `gym_classics2` tabular environment | Value array |
| Policy iteration | `dynamic_programming` | Yes | `gym_classics2` tabular environment | Policy array |
| MC prediction | `monte_carlo_methods` | No | Discrete | Value array |
| MC control ES | `monte_carlo_methods` | No | Discrete | Policy and Q arrays |
| Sarsa(0) | `temporal_difference_learning` | No | Discrete | Q array |
| Q-learning | `temporal_difference_learning` | No | Discrete | Q array |
| Semi-gradient TD/Sarsa | `linear_approximation` | No | Feature function | Weight array |
| Semi-gradient Sarsa($\lambda$) | `eligibility_traces` | No | Feature function | Weight array |
| REINFORCE / actor-critic | `policy_gradient_methods` | No | Feature function | Parameter arrays |

## Common conventions

- `discount` or `gamma` is the reward discount in `[0, 1]`.
- `n` is the number of sampled episodes for model-free algorithms.
- `alpha` and `epsilon` may accept a scalar or a schedule where documented.
- Randomized algorithms accept `rng`, either as a NumPy random generator or an
  integer seed.
- `history=True` retains learning curves and intermediate arrays. This is useful
  for teaching and plotting but consumes more memory.
- Tabular policies are arrays indexed by state; each value is an action ID.
- Q-functions are arrays shaped `(number_of_states, number_of_actions)`.

Model-free methods use only Gymnasium's `reset` and `step` APIs and can work with
compatible environments outside `gym_classics2`. Dynamic-programming methods
require the package-specific `model` method and an unwrapped environment.

## Reproducible results

Create a NumPy generator and pass it to each randomized algorithm. Seed the
environment separately because algorithm choices and environment transitions
use independent random-number generators.

```python
import gymnasium as gym
import numpy as np
import gym_classics2

from gym_classics2.algorithms.temporal_difference_learning import Q_learning
from gym_classics2.algorithms.policy import random_policy

gym_classics2.register()

seed = 42
rng = np.random.default_rng(seed)

env = gym.make("ClassicGridworld-v1", tabular=True)
env.reset(seed=seed)

Q = Q_learning(
    env,
    discount=0.99,
    alpha=0.1,
    epsilon=0.1,
    n=1_000,
    rng=rng,
)
```

Calling `env.reset(seed=seed)` initializes the environment's random stream.
Subsequent `env.reset()` calls continue that stream, so algorithms may start
new episodes without reseeding it. Passing the same seed on every reset instead
restarts the stream and may generate identical episode conditions.

Passing an integer directly is convenient for a single algorithm call:

```python
Q = Q_learning(env, discount=0.99, alpha=0.1, epsilon=0.1, rng=42)
```

For several calls, prefer one generator so they consume a single, reproducible
sequence:

```python
rng = np.random.default_rng(42)
policy = random_policy(env, rng=rng)
Q = Q_learning(env, discount=0.99, alpha=0.1, epsilon=0.1, rng=rng)
```

`np.random.seed(...)` does not seed `default_rng` and therefore does not control
these algorithms. If code calls `env.action_space.sample()` directly, seed that
space separately with `env.action_space.seed(seed)`, or sample discrete actions
with `rng.integers(env.action_space.n)`.
