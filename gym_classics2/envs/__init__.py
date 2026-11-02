"""Finite environments with a Gymnasium interface and explicit model access.

`gym_classics2` focuses on simple, finite environments and adds support for explicit model access and tabular representations.
The environments can be initialized as either tabular or non-tabular using the `tabular` parameter. Tabular versions represent states 
as integer IDs, while non-tabular versions use the original state representation. Tabular algorithms in `gym_classics2` require the tabular 
version of the environment, while non-tabular algorithms can work with either version and also regular Gymnasium environments with an 
appropriate state representation.

The environments are designed to be easy to understand and modify, making them suitable for educational purposes and research in reinforcement learning.
The abstract classes in ``envs.abstract`` provide reusable building blocks for defining
tasks:

- ``BaseEnv`` supplies the common interface, state encoding, and model access.
- ``LinearWalk`` defines a one-dimensional walk with terminal outcomes at both ends.
- ``Gridworld`` defines a deterministic gridworld from an ASCII layout.
- ``NoisyGridworld`` adds 80-10-10 stochastic action outcomes to ``Gridworld``.

Look at Concrete environments in ``envs.concrete`` for examples of how to use these building blocks to define specific tasks.
"""
