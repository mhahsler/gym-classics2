# Create a stochastic Gridworld

This tutorial recreates the 4×3 Classic Gridworld as a custom environment. The
agent intends to move in one direction, but the environment applies the
**80–10–10 rule**:

- The requested action is executed with probability 0.8.
- The action rotated 90 degrees counter-clockwise is executed with probability
  0.1.
- The action rotated 90 degrees clockwise is executed with probability 0.1.

For example, requesting `up` may move the agent up, left, or right. Walls,
blocks, and the edge of the grid cause the agent to remain in its current cell.

## Define the layout

A Gridworld layout is a rectangular string in which `S` marks a start, `G`
marks a terminal goal, `X` marks a blocked cell, and a space marks a traversable
cell. Coordinates start at `(0, 0)` in the lower-left corner.

```text
|   G|  y = 2: positive terminal at (3, 2)
| X G|  y = 1: block at (1, 1), negative terminal at (3, 1)
|S   |  y = 0: start at (0, 0)
```

Both terminal cells use `G` because the layout describes termination, while the
reward method below distinguishes their rewards.

## Implement the environment

Subclass `NoisyGridworld` to reuse its 80–10–10 sampling and exact transition
model. The custom class only needs to define its layout and reward behavior.

```python
from gym_classics2.envs.abstract.noisy_gridworld import NoisyGridworld


class StochasticClassicGridworld(NoisyGridworld):
    """The stochastic 4×3 Classic Gridworld."""

    layout = """
|   G|
| X G|
|S   |
"""

    def __init__(
        self,
        goal_reward=1.0,
        trap_reward=-1.0,
        step_reward=-0.04,
        **kwargs,
    ):
        self.trap_reward = trap_reward
        super().__init__(
            self.layout,
            goal_reward=goal_reward,
            step_reward=step_reward,
            **kwargs,
        )

    def _reward(self, state, action, next_state):
        # Once terminal, the absorbing transition has no additional reward.
        if state in self._goals:
            return 0.0

        if next_state == (3, 1):
            return self.trap_reward
        if next_state == (3, 2):
            return self._goal_reward
        return self._step_reward

    def _generate_transitions(self, state, action):
        # Planning algorithms may query terminal states, so make them absorbing.
        if state in self._goals:
            yield state, 0.0, True, 1.0
            return

        yield from super()._generate_transitions(state, action)
```

`Gridworld._done` already terminates transitions whose next state is a `G` cell,
so it does not need to be overridden.

## Run the custom environment

The class can be instantiated directly. Set `tabular=True` when using the
included tabular algorithms.

```python
env = StochasticClassicGridworld(tabular=True)

state, info = env.reset(seed=42)
action = env.action2id("up")
next_state, reward, terminated, truncated, info = env.step(action)

print("state:", env.id2state(state))
print("next state:", env.id2state(next_state))
env.close()
```

Calling `reset(seed=42)` also seeds the stochastic action selection because
`NoisyGridworld` samples through Gymnasium's `self.np_random` generator.

## Inspect the exact transition model

Sampling with `step` produces one outcome. The `model` method returns every
possible outcome and is what planning algorithms such as value iteration use.

```python
state = env.state2id((0, 0))
action = env.action2id("up")

next_states, rewards, terminals, probabilities = env.model(state, action)

for next_state, reward, terminal, probability in zip(
    next_states, rewards, terminals, probabilities
):
    print(env.id2state(next_state), reward, terminal, probability)
```

The output is:

```text
(0, 0) -0.04 False 0.1
(0, 1) -0.04 False 0.8
(1, 0) -0.04 False 0.1
```

The first outcome remains at `(0, 0)` because the unintended `left` action hits
the boundary.

## How the stochastic interface works

`NoisyGridworld` keeps simulation and planning consistent through three hooks:

| Hook | Used by | Responsibility |
| --- | --- | --- |
| `_sample_random_elements` | `step` | Sample the action actually executed |
| `_next_state` | `step` and `model` | Apply that action and report its probability |
| `_generate_transitions` | `model` | Enumerate all three possible actions |

When implementing a different probability distribution, update both sampling
and enumeration. Their outcomes and probabilities must agree, probabilities
must be nonnegative, and the probabilities returned for each state-action pair
must sum to one. Use `self.np_random` rather than the global `numpy.random`
generator so Gymnasium seeding remains reproducible.

## Optional Gymnasium registration

Direct construction is simplest during development. To use `gym.make`, register
the class once in the current Python process:

```python
import gymnasium as gym

gym.register(
    id="TutorialStochasticGridworld-v0",
    entry_point=StochasticClassicGridworld,
)

env = gym.make("TutorialStochasticGridworld-v0", tabular=True)
```

Use a unique ID to avoid colliding with an environment registered by another
package.
