# Dynamic programming

Value iteration and policy iteration operate on an exact finite-MDP model. Pass
the unwrapped, tabular environment to these functions.

## Value iteration

```python
import gymnasium as gym
import gym_classics2
from gym_classics2.algorithms.dynamic_programming import value_iteration

gym_classics2.register()
env = gym.make("ClassicGridworld-v1", tabular=True).unwrapped
values = value_iteration(env, discount=0.99, precision=1e-6)

env.print(values)
```

Set `history=True` to receive a list containing the initial values and the value
array after every sweep. This is convenient for animation.

## Policy iteration

```python
from gym_classics2.algorithms.dynamic_programming import policy_iteration

policy = policy_iteration(env, discount=0.99, precision=1e-6)
print(env.id2action(policy))
```

With `history=True`, policy iteration returns `(policy_history, value_history)`.
See the [dynamic-programming API](../api/algorithms.md#dynamic-programming) for
exact parameters and return values.
