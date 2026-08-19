# gym_classics2: Classic Discrete finite MDPs for Reinforcement Learning
[![license](https://img.shields.io/badge/license-GPL%20v3.0-blue)](./LICENSE)

`gym_classics2` is a collection of well-known discrete MDPs from the reinforcement learning
literature implemented as Gymnasium environments originally developed 
as [gym_classics](https://github.com/brett-daley/gym-classics) by Brett Daley. The library is intended to be used
in an introductory RL course like [Introduction to Reinforcement Learning:
Material for an introduction course to reinforcement learning for compute scientists].

`gym_classics2` provides:

* Expanded API to support discrete MDPs with model access. 
* Updated `gym_classics` environments to match Sutton and Barto's definition more closely.
* Support for maze visualization.
* Algorithms for popular model-based and model-free RL algorithms. 

## How to install

1. Install `Gymnasium` following the [Setup Gymnasium Notebook.](examples/Setup_Gymnasium.ipynb)
2. Install `gym_classics2` following the [Setup gym_classics2 Notebook.](examples/Setup_gym_classics2.ipynb)


## Examples

### Using the Standard Gymnasium Interface

```python
import gymnasium as gym
import gym_classics2
gym_classics2.register()  

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
import gym_classics2
gym_classics2.register()  

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

A complete and detailed example can be found in the following notebook:
[4x3 Grid World: Value and Policy Iteration.](examples/4x3_grid_world.ipynb)

## Documentation

[Manual pages](/examples/gym_classics2_help_pages.ipynb)


## References

* Brett Daley (2021), Gym Classics, GitHub repository, https://github.com/brett-daley/gym-classics
* Michael Hahsler (2025), Introduction to Reinforcement Learning: Material for an Introduction Course, GitHun repository, https://mhahsler.github.io/Introduction_to_Reinforcement_Learning/
* Sutton and Barto (2018), Reinforcement Learning: An Introduction, 2nd Ed., http://incompleteideas.net/book/the-book-2nd.html
* Mark Towers et al (2015), Gymnasium: A Standard Interface for Reinforcement Learning Environments, https://doi.org/10.48550/arXiv.2407.17032