"""Textbook-oriented reinforcement-learning algorithms and policy utilities.

The available families are grouped into submodules:

- ``dynamic_programming``: model-based policy evaluation, value iteration, and
  policy iteration.
- ``monte_carlo_methods``: prediction and exploring-starts control using sampled
  episodes.
- ``temporal_difference_learning``: tabular Sarsa(0) and Q-learning.
- ``linear_approximation``: semi-gradient TD(0) prediction and Sarsa(0) control
  with feature-based value functions.
- ``eligibility_traces``: semi-gradient Sarsa(lambda) control.
- ``policy_gradient_methods``: REINFORCE and actor-critic with parameterized
  policies.

The ``policy`` and ``schedules`` submodules provide policy helpers and schedules
for exploration rates and step sizes. Import functions from the specific
submodule that defines them, for example::

    from gym_classics2.algorithms.dynamic_programming import value_iteration
"""
