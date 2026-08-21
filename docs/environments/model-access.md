# Model access

Planning algorithms need the probability distribution
$p(s', r \mid s, a)$ rather than sampled transitions. Every registered
`gym_classics2` environment exposes this distribution through `model`.

Gymnasium wrappers intentionally expose only the standard interface, so unwrap
the environment first:

```python
import gymnasium as gym
import gym_classics2

gym_classics2.register()
env = gym.make("ClassicGridworld-v1", tabular=True).unwrapped
```

## Return value

```python
next_states, rewards, terminals, probabilities = env.model(state, action)
```

All four sequences have the same length; position `i` describes one possible
outcome.

| Value | Type | Meaning |
| --- | --- | --- |
| `next_states` | `list` | Possible successor IDs in tabular mode, raw states otherwise |
| `rewards` | 1-D `numpy.ndarray` | Reward for each successor |
| `terminals` | 1-D `numpy.ndarray[bool]` | Whether each transition terminates the episode |
| `probabilities` | 1-D `numpy.ndarray[float]` | Outcome probabilities, which sum to one |


**Note:** `model()` is cached by state-action pair. Treat returned values as read-only so
later calls continue to represent the environment correctly.

## Raw and tabular states

When `tabular=True`, pass an integer state ID to `model`. When `tabular=False`,
pass a raw state such as `(0, 0)`. Actions are integer IDs in both modes.

## Examples

### Get Transitions for a Stochastic Action

```python
state = env.state2id((0, 0))
action = env.action2id("up")
next_states, rewards, terminals, probabilities = env.model(state, action)

for next_state, reward, terminal, probability in zip(
    next_states, rewards, terminals, probabilities
):
    print(env.id2state(next_state), reward, terminal, probability)
```

Note that the state IDs are translated into state labels (i.e., coordinates) for easier readability.

```text
(0, 0) -0.04 False 0.1
(0, 1) -0.04 False 0.8
(1, 0) -0.04 False 0.1
```

The intended action occurs with probability 0.8. The actions rotated 90 degrees
to either side each occur with probability 0.1. In a deterministic environment,
the returned sequences contain one outcome with probability 1.

### Extract the Complete Transition Model as Matrices

We can construct complete transition model and reward matrices from the sparse transition information
provided by `model(s, a)`. Since there is one matrix per action, we construct a
transition tensor $P(s' \mid s, a)$ and the expected reward matrix $R(s, a)$.

```python
import numpy as np

n_states = env.observation_space.n
n_actions = env.action_space.n

# P[s, a, sp] is the probability of reaching sp from (s, a).
P = np.zeros((n_states, n_actions, n_states))

# R[s, a] is the expected immediate reward.
R = np.zeros((n_states, n_actions))

for s in env.states():
    for a in env.actions():
        next_states, rewards, terminals, probabilities = env.model(s, a)

        for sp, reward, terminal, probability in zip(
            next_states, rewards, terminals, probabilities
        ):
            P[s, a, sp] += probability
            R[s, a] += probability * reward

np.testing.assert_allclose(P.sum(axis=2), 1.0)

print(P.shape)
print(R.shape)
print(P.sum(axis=2)[0])
```

```text
(11, 4, 11)
(11, 4)
[1. 1. 1. 1.]
```

To extract the state-transition matrix for one action:

```python
up = env.action2id("up")
P_up = P[:, up, :]

print(P_up.shape)
print(P_up)
```

```text
(11, 11)
[[0.1 0.8 0.  0.1 0.  0.  0.  0.  0.  0.  0. ]
 [0.  0.2 0.8 0.  0.  0.  0.  0.  0.  0.  0. ]
 [0.  0.  0.9 0.  0.1 0.  0.  0.  0.  0.  0. ]
 [0.1 0.  0.  0.8 0.  0.1 0.  0.  0.  0.  0. ]
 [0.  0.  0.1 0.  0.8 0.  0.  0.1 0.  0.  0. ]
 [0.  0.  0.  0.1 0.  0.  0.8 0.  0.1 0.  0. ]
 [0.  0.  0.  0.  0.  0.  0.1 0.8 0.  0.1 0. ]
 [0.  0.  0.  0.  0.1 0.  0.  0.8 0.  0.  0.1]
 [0.  0.  0.  0.  0.  0.1 0.  0.  0.1 0.8 0. ]
 [0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0. ]
 [0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1. ]]
```

The reward matrix can be accessed in as similar way.

**Note:** This dense matrix representation becomes too large for 
problems with many states and actions.  
