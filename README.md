# gym_classics2

[![license](https://img.shields.io/badge/license-GPL%20v3.0-blue)](LICENSE)
[![documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://mhahsler.github.io/gym-classics2/)

`gym_classics2` provides classic finite Markov decision processes and readable
reinforcement-learning algorithms for teaching with
[Gymnasium](https://gymnasium.farama.org/). It is based on Brett Daley's
[gym-classics](https://github.com/brett-daley/gym-classics) and follows examples
from Sutton and Barto's *Reinforcement Learning: An Introduction*.

The package includes:

- Gymnasium environments for gridworlds, random walks, mazes, and cliff walking.
- Direct access to the transition and reward model, $p(s',r\mid s,a)$.
- Textbook-oriented implementations of dynamic programming, Monte Carlo,
  temporal-difference, function-approximation, eligibility-trace, and
  policy-gradient algorithms.
- Gridworld plotting and animation helpers.

## Installation

Python 3.9 or newer is required. To install the current development version:

```bash
python -m pip install "gym-classics2 @ git+https://github.com/mhahsler/gym-classics2.git"
```

For an editable checkout:

```bash
git clone https://github.com/mhahsler/gym-classics2.git
cd gym-classics2
python -m pip install -e .
```

Notebook tooling is optional:

```bash
python -m pip install -e ".[notebooks]"
```

## Quick start

Register the environments, create one with Gymnasium, and use the standard
`reset`/`step` interface:

```python
import gymnasium as gym
import gym_classics2

gym_classics2.register()
env = gym.make("ClassicGridworld-v1", tabular=True)

state, info = env.reset(seed=42)
for t in range(100):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(t, state, action, reward, next_state, done)
    state = next_state
    if done:
        state, info = env.reset()

env.close()
```

## Model access

The unwrapped environments expose the full finite-MDP model. In tabular mode,
states and next states are integer IDs:

```python
gym_env = gym.make("ClassicGridworld-v1", tabular=True)
env = gym_env.unwrapped  # expose the gym_classics2-specific API

print("Starts:", env.start_states)
print("Goals:", env.goal_states)

state = env.state2id((0, 0))
action = env.action2id("up")
next_states, rewards, terminals, probabilities = env.model(state, action)
```

For this transition, the result is equivalent to:

```text
next_states   = [0, 1, 3]
rewards       = [-0.04, -0.04, -0.04]
terminals     = [False, False, False]
probabilities = [0.1, 0.8, 0.1]
```

See the [model-access guide](https://mhahsler.github.io/gym-classics2/model-access/)
for raw states, array shapes, terminal transitions, and deterministic models.

## Included environments

| Gymnasium ID | Task | Dynamics | Default state representation |
| --- | --- | --- | --- |
| `5Walk-v0` | Five-state random walk | Deterministic | Tabular |
| `19Walk-v0` | Nineteen-state random walk | Deterministic | Tabular |
| `ClassicGridworld-v1` | 4×3 gridworld | Stochastic (80–10–10) | Tabular |
| `LMaze-v0` | L-shaped maze | Deterministic | Tabular |
| `CliffWalk-v1` | Cliff walking | Deterministic | Tabular |
| `DynaMaze-v0` | Dyna maze | Deterministic | Tabular |
| `FourRooms-v0` | Four rooms | Deterministic | Tabular |
| `SparseGridworld-v0` | Sparse-reward gridworld | Stochastic (80–10–10) | Tabular |
| `WindyGridworld-v0` | Windy gridworld | Deterministic | Tabular |

## Included algorithms

| Family | Implementations | Requires model access? |
| --- | --- | ---: |
| Dynamic programming | value iteration, policy iteration | Yes |
| Monte Carlo | prediction, exploring-starts control | No |
| Temporal difference | Sarsa(0), Q-learning | No |
| Linear approximation | semi-gradient TD(0), Sarsa(0), Fourier features | No |
| Eligibility traces | semi-gradient Sarsa($\lambda$) | No |
| Policy gradient | REINFORCE, actor-critic | No |

The model-free algorithms can also be used with compatible Gymnasium
environments outside this package. See the
[algorithm guide](https://mhahsler.github.io/gym-classics2/algorithms/overview/)
for input requirements and return values.

## Documentation and examples

- [Documentation](https://mhahsler.github.io/gym-classics2/)
- [4×3 Gridworld: value and policy iteration](examples/4x3_grid_world.ipynb)
- [Frozen Lake: Monte Carlo methods](examples/frozen_lake_MC.ipynb)
- [Course materials](https://mhahsler.github.io/Introduction_to_Reinforcement_Learning/)

To build the documentation locally:

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

## References

- Brett Daley (2021), [Gym Classics](https://github.com/brett-daley/gym-classics).
- Michael Hahsler (2025), [Introduction to Reinforcement Learning](https://mhahsler.github.io/Introduction_to_Reinforcement_Learning/).
- Richard S. Sutton and Andrew G. Barto (2018), [*Reinforcement Learning: An Introduction*, second edition](http://incompleteideas.net/book/the-book-2nd.html).
- Mark Towers et al. (2024), [Gymnasium: A Standard Interface for Reinforcement Learning Environments](https://doi.org/10.48550/arXiv.2407.17032).
