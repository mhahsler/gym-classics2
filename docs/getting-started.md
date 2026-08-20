# Getting started

## Install

`gym_classics2` requires Python 3.9 or newer. Install the current version from
GitHub:

```bash
python -m pip install "gym-classics2 @ git+https://github.com/mhahsler/gym-classics2.git"
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/mhahsler/gym-classics2.git
cd gym-classics2
python -m pip install -e .
```

## Run an environment

Registration connects the package's environment IDs to Gymnasium. It only
needs to happen once in a Python process.

```python
import gymnasium as gym
import gym_classics2

gym_classics2.register()
env = gym.make("ClassicGridworld-v1", tabular=True)

observation, info = env.reset(seed=42)
terminated = truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

Gymnasium returns both `terminated` (the task reached a terminal condition) and
`truncated` (an external time or step limit was reached). Treat the episode as
finished when either is true.

## Choose a state representation

Gridworld constructors accept `tabular=True` by default:

- `tabular=True` represents observations as consecutive integer IDs and is
  required by the tabular algorithms included in this package.
- `tabular=False` represents grid observations as `(x, y)` coordinates and is
  useful for visualization and function approximation.

The unwrapped environment converts between the two forms:

```python
base_env = env.unwrapped
state_id = base_env.state2id((0, 0))
coordinates = base_env.id2state(state_id)
```

Continue with [Environments](environments.md) or [Model access](model-access.md).
