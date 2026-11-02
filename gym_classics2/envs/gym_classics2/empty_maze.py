from numbers import Integral

from gym_classics2.envs.abstract.gridworld import Gridworld


class EmptyMazeGridworld(Gridworld):
    """A deterministic rectangular grid without interior walls.

    The agent starts in the lower-left corner (0, 0) and reaches the goal in
    the upper-right corner. By default, reaching the goal yields +1 and
    terminates the episode; other steps yield 0.

    Args:
        dims: Grid dimensions (width, height), defaulting to (4, 4). Dimensions
            must be positive integers with at least two cells in total.
        **args: Options passed to ``Gridworld``, including ``tabular``,
            ``render_mode``, ``goal_reward``, and ``step_reward``.
    """

    def __init__(self, dims=(4, 4), **args):
        width, height = dims
        if (not all(isinstance(size, Integral) and size > 0 for size in dims)
                or width * height < 2):
            raise ValueError("dims must be positive integers with at least two cells")

        rows = [[" "] * width for _ in range(height)]
        rows[-1][0] = "S"
        rows[0][-1] = "G"
        self.layout = "\n".join("|" + "".join(row) + "|" for row in rows)
        super().__init__(self.layout, **args)
