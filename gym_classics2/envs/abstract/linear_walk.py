from gym_classics2.envs.abstract.base_env import BaseEnv


class LinearWalk(BaseEnv):
    """Finite one-dimensional walk with terminal outcomes at both ends.

    Raw states are integer positions from ``0`` through ``length - 1``. Every
    episode starts at the center position. Action ``0`` moves left and action ``1``
    moves right; movement is deterministic. Attempting to move left from position
    ``0`` or right from position ``length - 1`` terminates the episode and yields
    the corresponding boundary reward. All other transitions have reward zero.

    Observations are consecutive integer state IDs. For this environment, each ID
    is equal to its raw position.

    Args:
        length: Odd number of nonterminal positions in the walk.
        left_reward: Reward for terminating beyond the left boundary.
        right_reward: Reward for terminating beyond the right boundary.
    """

    def __init__(self, length, left_reward, right_reward):
        self._length = length
        self._left_reward = left_reward
        self._right_reward = right_reward

        assert length % 2 == 1
        super().__init__(starts={length // 2}, action_labels=["left", "right"])

    def _next_state(self, state, action):
        state += [-1, 1][action]
        next_state = min(max(state, 0), self._length - 1)
        return next_state, 1.0

    def _reward(self, state, action, next_state):
        sa_pair = (state, action)
        return {
            (0, 0):                self._left_reward,
            (self._length - 1, 1): self._right_reward,
        }.get(sa_pair, 0.0)

    def _done(self, state, action, next_state):
        sa_pair = (state, action)
        return sa_pair in {(0, 0), (self._length - 1, 1)}

    def _generate_transitions(self, state, action):
        yield self._deterministic_step(state, action)
