from gym_classics2.envs.abstract.gridworld import Gridworld

class FourRooms(Gridworld):
    """An 11x11 gridworld segmented into four rooms. The agent begins in the bottom-left
    cell; the goal is in the top-right cell.

    **Reference:** Sutton, Precup, and Singh,
    [*Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in
    Reinforcement Learning* (1999)](https://doi.org/10.1016/S0004-3702%2899%2900052-1),
    p. 192.

    **state**: Grid location.

    **actions**: Move up/right/down/left.

    **rewards**: +1 for episode termination.

    **termination**: Taking any action in the goal.
    """

    layout = """
|     X     |
|     X   G |
|           |
|     X     |
|     X     |
|X XXXX     |
|     XXX XX|
|     X     |
|     X     |
|           |
|S    X     |
"""

    def __init__(self, **args):
        super().__init__(FourRooms.layout, **args)
