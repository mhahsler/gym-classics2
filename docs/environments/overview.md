# Environments

Call `gym_classics2.register()` before using these IDs with `gymnasium.make`.
All registered environments have discrete action spaces and support the model
interface described in [Model access](model-access.md).

| Gymnasium ID | Description | Dynamics | Sutton and Barto text book |
| --- | --- | --- | --- |
| `5Walk-v0` | Five nonterminal states between two terminal outcomes | Deterministic | Example 6.2 |
| `19Walk-v0` | Longer random-walk task with rewards −1 and +1 | Deterministic | Example 7.1 |
| `ClassicGridworld-v1` | 4×3 grid with a wall, goal, and trap | 80–10–10 action noise | Exercise 3.14-style gridworld |
| `LMaze-v0` | Maze with an L-shaped barrier | Deterministic | - |
| `CliffWalk-v1` | 12×4 cliff-walking control task | Deterministic | Example 6.6 |
| `DynaMaze-v0` | 9×6 maze used to demonstrate planning | Deterministic | Example 8.1 |
| `FourRooms-v0` | 11×11 environment divided by doorways | Deterministic | - |
| `SparseGridworld-v0` | Open grid with sparse rewards | 80–10–10 action noise | Figure 7.4 |
| `WindyGridworld-v0` | Column-dependent upward wind | Deterministic | Example 6.5 |

## Common constructor options

The gridworld environments accept these keyword arguments unless a subclass
documents a different default:

| Option | Meaning |
| --- | --- |
| `tabular` | Return integer state IDs (`True`) or raw coordinates (`False`) |
| `render_mode` | `None`, `"human"`, or `"rgb_array"` |
| `goal_reward` | Reward associated with reaching a goal |
| `step_reward` | Reward for an ordinary transition |

`ClassicGridworld-v1` additionally accepts `trap_reward`.
`CliffWalk-v1` additionally accepts `cliff_reward`.

## Coordinates and actions

Grid coordinates are `(x, y)` tuples with `(0, 0)` at the lower-left. The
default actions are integer IDs corresponding to `up`, `right`, `down`, and
`left`. Convert between representations with `state2id`, `id2state`,
`action2id`, and `id2action`.

Start and goal coordinates are available without accessing private fields:

```python
env = gym.make("ClassicGridworld-v1").unwrapped
print(env.start_states)
print(env.goal_states)
```
