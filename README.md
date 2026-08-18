# Heavily Modified Version of Gym Classics
[![license](https://img.shields.io/badge/license-GPL%20v3.0-blue)](./LICENSE)

Gym Classics is a collection of well-known discrete MDPs from the reinforcement learning
literature implemented as OpenAI Gym environments originally developed by [Brett Daley](https://github.com/brett-daley/gym-classics).


This version includes:

* Updated environments to match Sutton and Barto's definition more closely.
* Support for maze visualization.
* Extra algorithms.
* More utility functions.

## How to install

From the shell: activate the Python environment you want to use and execute the following: 

```
git clone https://github.com/mhahsler/gym-classics.git
cd gym-classics
git pull
pip install -e .
```

In Colab and Jupyter notebooks, you can add a block with the following code block:

```
!git clone https://github.com/mhahsler/gym-classics.git
!cd gym-classics;git pull
!cd gym-classics; pip install -e .
```

## Examples

### Using the Standard Gymnasium Interface

```python
import gymnasium as gym
import gym_classics
gym_classics.register()  

env = gym.make('ClassicGridworld-v1', tabular = True)

for t in range(1, 100 + 1):
    action = env.action_space.sample()  # Select a random action
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print("t={}, state={}, action={}, reward={}, next_state={}, done={}".format(
        t, state, action, reward, next_state, done))
    if done:
        next_state, _ = env.reset()
    state = next_state

env.close()
```

### Using Model Access

```python
import gymnasium as gym
import gym_classics
gym_classics.register()  

gym_env = gym.make('ClassicGridworld-v1', tabular = True)

### unwrapping is necessary to expose the model access interface
env = gym_env.unwrapped

print(f"States {env.states()}: {[env.id2state(s) for s in env.states()]}")
print(f"Actions {env.actions()}: {[env.id2action(a) for a in env.actions()]}")
print("Start states:", env._starts)
print("Goals (terminal states):", env._goals)
```

```
States range(0, 11): [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
Actions range(0, 4): ['up', 'right', 'down', 'left']
Start states: [(0, 0)]
Goals (terminal states): {(3, 1), (3, 2)}
```

The model method implements access to the transition and reward model $P(s', r | s, a)$. 
We specify $s$ and $a$ and get a table for all possible $s'$, with the reward $r$, the information if $s'$ 
is a terminal state, and the transition probability.

```python
print("Model for state (0,0) and action up):\n(next_states, rewards, terminals, probs)\n")
display(env.model(env.state2id((0,0)), env.action2id("up")))
```

```
Model for state (0,0) and action up):
(next_states, rewards, terminals, probs)

[[0, 1, 3],
 array([-0.04, -0.04, -0.04]),
 array([False, False, False]),
 array([0.1, 0.8, 0.1])]
```

Look at the [notebook with a detailed code example.](common/gym-classics/examples/4x3_grid_world.ipynb)
