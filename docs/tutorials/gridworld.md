# Solve Classic Gridworld

This tutorial combines exact model access, value iteration, and a greedy policy.

```python
import gymnasium as gym
import gym_classics2
from gym_classics2.algorithms.dynamic_programming import value_iteration
from gym_classics2.algorithms.policy import greedy_policy

gym_classics2.register()
env = gym.make("ClassicGridworld-v1", tabular=True).unwrapped

values = value_iteration(env, discount=0.99, precision=1e-6)
policy = greedy_policy(env, values, discount=0.99)

env.print(values)
env.print(env.id2action(policy))
```

The values quantify expected discounted return from each reachable cell. The
policy applies a one-step Bellman look-ahead to those values and chooses a
maximizing action; ties are broken randomly.

For a notebook with visualizations and intermediate sweeps, open
[`examples/4x3_grid_world.ipynb`](https://github.com/mhahsler/gym-classics2/blob/main/examples/4x3_grid_world.ipynb).
