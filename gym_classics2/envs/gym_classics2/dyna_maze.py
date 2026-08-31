from gym_classics2.envs.abstract.gridworld import Gridworld


class DynaMaze(Gridworld):
    """A 9x6 deterministic gridworld with barriers to make navigation more challenging.
    The agent starts in cell (0, 3); the goal is the top-right cell.

    **Reference:** Sutton and Barto,
    [*Reinforcement Learning: An Introduction* (2nd ed., 2018)](https://incompleteideas.net/book/the-book-2nd.html),
    p. 164, Example 8.1.

    **state**: Grid location.

    **actions**: Move up/right/down/left.

    **rewards**: +1 for episode termination.

    **termination**: Reaching the goal.
    """

    layout = """
|       XG|
|  X    X |
|S X    X |
|  X      |
|     X   |
|         |
"""

    def __init__(self, **args):
        super().__init__(DynaMaze.layout, **args)
