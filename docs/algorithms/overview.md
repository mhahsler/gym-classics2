# Choosing an algorithm

The included implementations are organized by the information available to the
agent and by the form of the value function.

| Algorithm | Module | Model? | State-space requirement | Main result |
| --- | --- | ---: | --- | --- |
| Value iteration | `dynamic_programming` | Yes | `gym_classics2` tabular environment | Value array |
| Policy iteration | `dynamic_programming` | Yes | `gym_classics2` tabular environment | Policy array |
| MC prediction | `monte_carlo_methods` | No | Discrete | Value array |
| MC control ES | `monte_carlo_methods` | No | Discrete | Policy and Q arrays |
| Sarsa(0) | `temporal_difference_learning` | No | Discrete | Q array |
| Q-learning | `temporal_difference_learning` | No | Discrete | Q array |
| Semi-gradient TD/Sarsa | `linear_approximation` | No | Feature function | Weight array |
| Semi-gradient Sarsa($\lambda$) | `eligibility_traces` | No | Feature function | Weight array |
| REINFORCE / actor-critic | `policy_gradient_methods` | No | Feature function | Parameter arrays |

## Common conventions

- `discount` or `gamma` is the reward discount in `[0, 1]`.
- `n` is the number of sampled episodes for model-free algorithms.
- `alpha` and `epsilon` may accept a scalar or a schedule where documented.
- `history=True` retains learning curves and intermediate arrays. This is useful
  for teaching and plotting but consumes more memory.
- Tabular policies are arrays indexed by state; each value is an action ID.
- Q-functions are arrays shaped `(number_of_states, number_of_actions)`.

Model-free methods use only Gymnasium's `reset` and `step` APIs and can work with
compatible environments outside `gym_classics2`. Dynamic-programming methods
require the package-specific `model` method and an unwrapped environment.
