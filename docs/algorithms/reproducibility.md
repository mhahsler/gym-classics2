# Reproducible results

Environments and algorithms each have their own random-number stream.

- **Stochastic environments:** Set the seed for the environment random-number
  generator during the first reset with `env.reset(seed=42)`. Subsequent
  `env.reset()` calls continue that stream, so algorithms may start new episodes
  without reseeding it.

- **Randomized algorithms:** Algorithms that sample actions or otherwise use
  randomization should have their own random-number stream. All randomized
  `gym_classics2` algorithms accept an `rng` parameter. Create a NumPy generator
  with `rng = np.random.default_rng(seed)` and pass it to the algorithm.

Here is an example:

```python
import gymnasium as gym
import numpy as np
import gym_classics2

from gym_classics2.algorithms.temporal_difference_learning import Q_learning

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

`env.reset(seed=seed)` initializes the environment's random stream, while
`np.random.default_rng(seed)` initializes the algorithm's random stream. Setting
both seeds before an experiment makes its behavior reproducible.

For experiments with stochastic environments or algorithms, run the algorithm
multiple times with different seeds and report the mean and standard deviation.

## Notes

- `np.random.seed(...)` does not seed `default_rng` and therefore does not
  control these algorithms.
- Environment methods can use the environment's generator through
  `self.np_random.choice()` and similar methods.
- If code calls `env.action_space.sample()` directly, seed that space separately
  with `env.action_space.seed(seed)`, or sample discrete actions with
  `rng.integers(env.action_space.n)`.
