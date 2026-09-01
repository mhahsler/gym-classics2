"""Scalar schedules for reinforcement-learning hyperparameters.

A schedule is a callable that maps a nonnegative episode or step index ``t`` to
a floating-point value. The included algorithms use schedules to vary parameters
such as the step size (alpha) and exploration rate (epsilon) during training.

For example, a linear schedule can decrease epsilon from 1.0 to 0.1 over
10 steps:

```python
schedule = LinearDecaySchedule(1.0, min_value=0.1, decay_steps=10)
for t in (0, 5, 10, 15):
    print(t, schedule(t))
# 0 1.0
# 5 0.55
# 10 0.1
# 15 0.1
```
"""

import numpy as np

class Schedule:
    """Interface for a scalar value indexed by episode or time step."""

    def __call__(self, t):
        """Evaluate the schedule.

        Args:
            t: Nonnegative episode or step index.

        Returns:
            Scheduled scalar value at ``t``.
        """
        raise NotImplementedError

class ConstantSchedule(Schedule):
    """Return the same value for every index.

    Args:
        value: Constant value returned by the schedule.
    """
    def __init__(self, value):
        self.value = float(value)

    def __call__(self, t):
        return self.value

class StepSchedule(Schedule):
    """Switch once from a high value to a low value.

    The schedule returns ``high_value`` while ``t < steps`` and ``low_value``
    from ``t == steps`` onward.

    Args:
        high_value: Value before the switch.
        low_value: Value at and after the switch.
        steps: Index at which to switch to ``low_value``.
    """
    def __init__(self, high_value, low_value, steps):
        self.high_value = float(high_value)
        self.low_value = float(low_value)
        self.steps = int(steps)

    def __call__(self, t):
        if t < self.steps:
            return self.high_value
        return self.low_value

class LinearDecaySchedule(Schedule):
    """Interpolate linearly from an initial value to a final value.

    The schedule returns ``initial_value`` at ``t = 0``, changes linearly through
    ``decay_steps``, and returns ``min_value`` thereafter.

    Args:
        initial_value: Value at index zero.
        min_value: Final value and lower endpoint of a decreasing schedule.
        decay_steps: Number of indices over which to interpolate.
    """
    def __init__(self, initial_value, min_value, decay_steps):
        self.initial_value = float(initial_value)
        self.min_value = float(min_value)
        self.decay_steps = int(decay_steps)

    def __call__(self, t):
        fraction = min(float(t) / max(1, self.decay_steps), 1.0)
        return self.initial_value + fraction * (self.min_value - self.initial_value)

class ExponentialDecaySchedule(Schedule):
    """Decay geometrically to a minimum value.

    At index ``t``, the value is ``max(min_value, initial_value * decay_rate**t)``.

    Args:
        initial_value: Unclipped value at index zero.
        min_value: Lower bound for the returned value.
        decay_rate: Multiplicative factor applied at each successive index.
    """
    def __init__(self, initial_value, min_value, decay_rate):
        self.initial_value = float(initial_value)
        self.min_value = float(min_value)
        self.decay_rate = float(decay_rate)

    def __call__(self, t):
        return max(self.min_value, self.initial_value * (self.decay_rate ** t))

class InverseDecaySchedule(Schedule):
    """Decay in inverse proportion to the index.

    The schedule returns ``initial_value`` at ``t = 0``. For ``t > 0``, it
    returns ``max(min_value, initial_value / t)``.

    Args:
        initial_value: Value at index zero and numerator of the inverse schedule.
        min_value: Lower bound for the returned value.
    """
    def __init__(self, initial_value, min_value=0.0):
        self.initial_value = float(initial_value)
        self.min_value = float(min_value)

    def __call__(self, t):
        if t == 0:
            return self.initial_value
        return max(self.min_value, self.initial_value / t)

def plot_schedule(schedule, steps=1000):
    """Plot scheduled values for indices ``0`` through ``steps - 1``.
    
    Example:
        ```python
        from gym_classics2.algorithms.schedules import (
            LinearDecaySchedule,
            plot_schedule,
        )

        epsilon = LinearDecaySchedule(1.0, min_value=0.1, decay_steps=1_000)
        plot_schedule(epsilon, steps=1_000)
        ```

    Args:
        schedule: Callable that accepts an integer index and returns a scalar.
        steps: Number of scheduled values to plot.
    """
    import matplotlib.pyplot as plt
    
    values = [schedule(t) for t in range(steps)]
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(steps), values, linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Schedule Value')
    plt.title('Schedule Plot')
    plt.grid(True)
    plt.show()
