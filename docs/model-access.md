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

## Example: a stochastic action

```python
state = env.state2id((0, 0))
action = env.action2id("up")
next_states, rewards, terminals, probabilities = env.model(state, action)

for next_state, reward, terminal, probability in zip(
    next_states, rewards, terminals, probabilities
):
    print(env.id2state(next_state), reward, terminal, probability)
```

The intended action occurs with probability 0.8. The actions rotated 90 degrees
to either side each occur with probability 0.1. In a deterministic environment,
the returned sequences contain one outcome with probability 1.

## Terminal transitions

`terminals[i]` describes the transition into `next_states[i]`. Algorithms should
not bootstrap from a terminal successor. The package's Bellman backup therefore
uses a multiplier equivalent to `1 - terminals`.

## Raw and tabular states

When `tabular=True`, pass an integer state ID to `model`. When `tabular=False`,
pass a raw state such as `(0, 0)`. Actions are integer IDs in both modes.

The model is cached by state-action pair. Treat returned values as read-only so
later calls continue to represent the environment correctly.
