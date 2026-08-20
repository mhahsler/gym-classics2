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
current state \(s\) and requested action \(a\). The notation
\(p(s,a,r,s')\) instead describes a joint distribution and additionally
requires a distribution over the state-action pair. If that distribution is
\(d(s,a)\), then

\[
p(s,a,r,s') = d(s,a)\,p(s',r\mid s,a).
\]

An environment supplies the conditional model \(p(s',r\mid s,a)\). A policy and
state-visitation process determine \(d(s,a)\).

For the 80–10–10 dynamics, introduce the action actually executed by the
environment, \(\tilde a\). Movement is deterministic after \(\tilde a\) is
known:

\[
p(s',r\mid s,a) =
\sum_{\tilde a}
p(\tilde a\mid a)\,
\mathbf{1}\{s'=f(s,\tilde a)\}\,
\mathbf{1}\{r=R(s,a,s')\}.
\]

Here, \(p(\tilde a\mid a)\) is the 80–10–10 action noise, \(f\) applies
movement and walls, and \(R\) computes the reward. The code below implements
each of these pieces.

## Define the layout

A Gridworld layout is a rectangular string in which `S` marks a start, `G`
marks a terminal goal, `X` marks a blocked cell, and a space marks a traversable
cell. Coordinates start at `(0, 0)` in the lower-left corner.

```text
|   G|  y = 2: positive terminal at (3, 2)
| X G|  y = 1: block at (1, 1), negative terminal at (3, 1)
|S   |  y = 0: start at (0, 0)
```

Both terminal cells use `G` because the layout describes termination. The
reward function distinguishes the positive goal from the negative trap.

## Implement the environment from `Gridworld`

The complete class explicitly implements both ways the dynamics are used:

- `step()` samples one executed action.
- `model()` enumerates every possible executed action.

```python
from gym_classics2.envs.abstract.gridworld import Gridworld


class StochasticClassicGridworld(Gridworld):
    """The stochastic 4×3 Classic Gridworld."""

    layout = """
|   G|
| X G|
|S   |
"""

    # Offsets use Gridworld's action order: up, right, down, left.
    # -1 is a counter-clockwise turn and +1 is a clockwise turn.
    action_offsets = (-1, 0, 1)
    action_probabilities = (0.1, 0.8, 0.1)

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

    def _sample_random_elements(self, state, action):
        """Sample the action that the environment actually executes."""
        offset = self.np_random.choice(
            self.action_offsets,
            p=self.action_probabilities,
        )
        executed_action = (action + int(offset)) % self.action_space.n
        return (executed_action,)

    def _next_state(self, state, action, executed_action):
        """Apply an executed action and return its conditional probability."""
        next_state, _ = super()._next_state(state, executed_action)
        probability = 0.8 if executed_action == action else 0.1
        return next_state, probability

    def _reward(self, state, action, next_state):
        """Return R(s, a, s')."""
        if state in self._goals:
            return 0.0
        if next_state == (3, 1):
            return self.trap_reward
        if next_state == (3, 2):
            return self._goal_reward
        return self._step_reward

    def _done(self, state, action, next_state):
        """Mark transitions into either G cell as terminal."""
        return next_state in self._goals

    def _generate_transitions(self, state, action):
        """Enumerate every outcome with nonzero p(s', r | s, a)."""
        if state in self._goals:
            # Planning algorithms may query terminal states. Make them absorbing.
            yield state, 0.0, True, 1.0
            return

        for offset in self.action_offsets:
            executed_action = (action + offset) % self.action_space.n
            yield self._deterministic_step(
                state,
                action,
                executed_action,
            )
```

## How the methods construct \(p(s',r\mid s,a)\)

`Gridworld` supplies deterministic movement, coordinate clamping, and blocked
cell handling. The subclass supplies the random action and reward model.

| Method | Model component | Role |
| --- | --- | --- |
| `_sample_random_elements` | Samples \(p(\tilde a\mid a)\) | Chooses one executed action when `step()` is called |
| `Gridworld._next_state` | \(f(s,\tilde a)\) | Applies movement, boundaries, and blocked cells |
| `_next_state` | \(s'\) and \(p(\tilde a\mid a)\) | Returns the resulting state and probability of that random event |
| `_reward` | \(R(s,a,s')\) | Assigns the reward associated with the transition |
| `_done` | Terminal indicator | Identifies transitions after which no future reward is available |
| `_generate_transitions` | Full \(p(s',r\mid s,a)\) | Enumerates all random events for planning algorithms |

For each enumerated executed action, the inherited `_deterministic_step` helper
calls `_next_state`, `_reward`, and `_done`, producing

```text
(next_state, reward, terminal, probability)
```

The public `model(state, action)` method collects these tuples into four
parallel sequences and checks that the probabilities are nonnegative and sum to
one.

!!! note "Different random events can produce the same next state"

    At a boundary, both `left` and `down` might leave the agent in the same
    cell. The model can contain separate rows for those random events. To obtain
    a single value for \(p(s',r\mid s,a)\), sum the probabilities of rows with
    identical `(next_state, reward)` values.

## Sample a transition with `step()`

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

`step()` calls `_sample_random_elements` once, then conditions the transition on
that sampled `executed_action`. Using `self.np_random` makes the result
reproducible after `reset(seed=...)`.

## Enumerate the model with `model()`

Sampling produces one transition. Planning algorithms such as value iteration
need every possible transition and call `model()` instead.

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

For this state and action, these rows are precisely the nonzero entries of
\(p(s',r\mid s=(0,0),a=\text{up})\). The first outcome stays at `(0, 0)`
because the unintended `left` action hits the boundary.

## Changing the stochastic dynamics

To implement a different noise distribution, change
`action_offsets`, `action_probabilities`, and the probability returned by
`_next_state`. Sampling and enumeration must describe the same distribution:

- Every outcome sampled by `_sample_random_elements` must be present in
  `_generate_transitions`.
- Their corresponding probabilities must match.
- Probabilities must be nonnegative and sum to one for each `(state, action)`.
- Random sampling should use `self.np_random` so Gymnasium seeding works.

For probabilities that are not simply 0.8 or 0.1, store the sampled offset—or
its probability—along with the executed action so `_next_state` can return the
correct value.

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
