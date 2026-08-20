from gym_classics2.envs.abstract.gridworld import Gridworld

class LMazeGridworld(Gridworld):
    """A deterministic 10x10 maze separated by an L-shaped barrier.

    The agent begins below the horizontal barrier and must travel around it to
    reach the goal near the upper-right corner.
    """
    layout = """
|          |
|        G |
|          |
| XXXXXX   |
|      X   |
|      X   |
|      X   |
|   S  X   |
|          |
|          |
"""

    def __init__(self, **args):
        super().__init__(self.layout, **args)
