# Monte Carlo control on Frozen Lake

The model-free algorithms are compatible with external Gymnasium environments
that have the required discrete spaces.

```python
import gymnasium as gym
from gym_classics2.algorithms.monte_carlo_methods import MC_control_ES

env = gym.make("FrozenLake-v1")
policy, q_values = MC_control_ES(
    env,
    discount=0.99,
    n=10_000,
    max_episode_len=100,
)
env.close()
```

Exploring starts require setting the initial state. Some third-party Gymnasium
environments or wrappers may not permit that operation; for those environments,
choose an on-policy or temporal-difference control method instead.

The complete notebook is
[`examples/frozen_lake_MC.ipynb`](https://github.com/mhahsler/gym-classics2/blob/main/examples/frozen_lake_MC.ipynb).
