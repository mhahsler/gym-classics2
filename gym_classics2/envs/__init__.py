"""Finite environments with a Gymnasium interface and explicit model access.

The classes in ``envs.abstract`` provide reusable building blocks for defining
tasks:

- ``BaseEnv`` supplies the common interface, state encoding, and model access.
- ``LinearWalk`` defines a one-dimensional walk with terminal outcomes at both ends.
- ``Gridworld`` defines a deterministic gridworld from an ASCII layout.
- ``NoisyGridworld`` adds 80-10-10 stochastic action outcomes to ``Gridworld``.

These classes describe general environment families. Subclass or configure them
to define a particular task. Concrete implementations in ``envs.gym_classics2``
choose layouts, rewards, and task-specific dynamics, such as ``Walk5``,
``ClassicGridworld``, and ``CliffWalk``. Importing ``gym_classics2`` registers the
supported concrete environments for creation with ``gymnasium.make``.
"""
