from gym_classics2.envs.abstract.gridworld import Gridworld

class FourRooms(Gridworld):
    """An 11x11 gridworld segmented into four rooms. The agent begins in the bottom-left
    cell; the goal is in the top-right cell.

    **reference:** cite{2} (page 192).

    **state**: Grid location.

    **actions**: Move up/right/down/left.

    **rewards**: +1 for episode termination.

    **termination**: Taking any action in the goal.
    
    Reference: Sutton, Precup and Singh: Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. 
        Artificial Intelligence, 112(1-2):181-211, 1999. [https://hdl.handle.net/20.500.14394/9879] 
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