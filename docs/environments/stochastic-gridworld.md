# Create a stochastic Gridworld

This tutorial implements the stochastic dynamics of the 4×3 Classic Gridworld
directly from `Gridworld`. The agent requests an action, but the environment
applies the **80–10–10 rule**:

- Execute the requested action with probability 0.8.
- Execute the action rotated 90 degrees counter-clockwise with probability 0.1.
- Execute the action rotated 90 degrees clockwise with probability 0.1.

Thus, requesting `up` can move the agent up, left, or right. A movement into a
wall, a blocked cell, or the edge of the grid leaves the agent in its current
cell.

## The transition and reward model

A finite Markov decision process is commonly specified by

\[
p(s', r \mid s, a),
\]

the probability of reaching next state \(s'\) and receiving reward \(r\), given
current state \(s\) and requested action \(a\). This is typically implemented
as a function \(p(s,a,r,s') \doteq p(s', r \mid s, a)\).

## Define the layout

A Gridworld layout is a rectangular string in which `S` marks a start, `G`
marks a terminal goal, `X` marks a blocked cell, and a space marks a traversable
cell. Coordinates start at `(0, 0)` in the lower-left corner.

```text
|   G|
| X G|
|S   |
```

Both terminal cells use `G` because the layout describes termination. The
reward function distinguishes the positive goal with a reward of +1 (top-right
corner) from the negative trap with a reward of -1 directly below it.

## Implement the environment from `Gridworld`

The `gym_classics2` base class provides two public interfaces to the dynamics:

- `model()` enumerates every possible outcome for a state-action pair.
- `step()` samples and executes one outcome.

The two interfaces use the following implementation methods:

| Method | Model component | Role |
| --- | --- | --- |
| `_sample_random_elements` | Samples \(p(\tilde a\mid a)\) | Chooses one executed action when `step()` is called |
| `_next_state` | \(s'\) and \(p(\tilde a\mid a)\) | Returns the resulting state and probability of that random event |
| `_reward` | \(R(s,a,s')\) | Assigns the reward associated with the transition |
| `_done` | Terminal indicator | Identifies transitions after which no future reward is available |
| `_generate_transitions` | Full \(p(s',r\mid s,a)\) | Enumerates all random events for planning algorithms |


### Step interface

The standard Gymnasium `step()` interface calls the implementation method in the following order:

1. `_sample_random_elements`
2. `_next_state`
3. `_reward`
4. `_done`

### Model interface

`model(state, action)` collects tuples from `_generate_transitions` into four
parallel sequences and checks that the probabilities are nonnegative and sum to
one.

!!! note "Different random events can produce the same next state"

    At a boundary, both `left` and `down` might leave the agent in the same
    cell. The model can contain separate rows for those random events. To obtain
    a single value for \(p(s',r\mid s,a)\), sum the probabilities of rows with
    identical `(next_state, reward)` values.

### Example

```python
import gymnasium as gym

from gym_classics2.envs.abstract.gridworld import Gridworld


class StochasticClassicGridworld(Gridworld):
    """The stochastic 4×3 Classic Gridworld."""

    layout = """
|   G|
| X G|
|S   |
"""

    def __init__(self, tabular=True):
        super().__init__(self.layout, tabular=tabular)

    # Used by the environment at the beginning of the step function to determine value of all random events.
    # Here, the random event is that the environment executes a potentially different
    # noisy action instead of the action the agent asked for.
    # Note: self.np_random.choice uses the environment's random number generator.
    # We return the actually executed noisy action for the step as a list of random elements.
    def _sample_random_elements(self, state, action):
        offset = self.np_random.choice([-1, 0, 1], p=[0.1, 0.8, 0.1])
        noisy_action = (action + int(offset)) % self.action_space.n
        return [noisy_action]

    # Returns the next state and the probability for the transition. Action is the agent's chosen action.
    # Noisy action is the actual randomized action that was sampled in _sample_random_elements and is executed.
    def _next_state(self, state, action, noisy_action):
        next_state, _ = super()._next_state(state, noisy_action)
        p = 0.8 if action == noisy_action else 0.1
        return next_state, p

    # Reward model
    def _reward(self, state, action, next_state):
        if state in self._goals:
            return 0.0
        return {(3, 1): -1.0, (3, 2): 1.0}.get(next_state, -0.04)

    # Terminal state indicator
    def _done(self, state, action, next_state):
        return next_state in self._goals

    # Returns an iterator for all possible outcomes.This function is used for model access.
    # The random element is that we have a noisy action, that may not be the intended action.
    # Yields: elements with structure (next_state, reward, done, prob)
    def _generate_transitions(self, state, action):
        # goal state is absorbing
        if state in self._goals:
            yield state, 0, True, 1.0

        else:
            for i in [-1, 0, 1]:
                noisy_action = (action + i) % self.action_space.n
                yield self._deterministic_step(state, action, noisy_action)

gym.register(
    id="TutorialStochasticGridworld-v0",
    entry_point=StochasticClassicGridworld,
)

print(gym.spec("TutorialStochasticGridworld-v0").id)
```

```text
TutorialStochasticGridworld-v0
```

#### Sample a transition with `step()`

The registered class can be instantiated with `gym.make`. Set `tabular=True`
when using the included tabular algorithms. Unwrap the environment to access
the package-specific model interface.

```python
env = gym.make("TutorialStochasticGridworld-v0", tabular=True).unwrapped

state, info = env.reset(seed=42)
action = env.action2id("up")
next_state, reward, terminated, truncated, info = env.step(action)

print("state:", env.id2state(state))
print("next state:", env.id2state(next_state))
print("reward:", reward)
print("terminated:", terminated)
print("truncated:", truncated)
```

```text
state: (0, 0)
next state: (0, 1)
reward: -0.04
terminated: False
truncated: False
```

#### Access the model with `model()`

Produce all possible transitions from a given state for a given action.

```python
state = env.state2id((0, 0))
action = env.action2id("up")

next_states, rewards, terminals, probabilities = env.model(state, action)

for next_state, reward, terminal, probability in zip(
    next_states, rewards, terminals, probabilities
):
    print(env.id2state(next_state), reward, terminal, probability)

env.close()
```

The output is:

```text
(0, 0) -0.04 False 0.1
(0, 1) -0.04 False 0.8
(1, 0) -0.04 False 0.1
```

For this state and action, these rows are precisely the nonzero entries of
\(p(s',r\mid s=(0,0),a=\text{up})\). The first outcome stays at `(0, 0)`
because the unintended `left` action hits the boundary.
