from gym_classics2.envs.abstract.gridworld import Gridworld

class NoisyGridworld(Gridworld):
    """Gridworld with classic 80-10-10 stochastic action outcomes.

    The requested action is executed with probability 0.8. With probability 0.1
    each, it is instead rotated 90 degrees clockwise or counterclockwise. The
    resulting move follows the boundary, blocking, reward, and termination rules
    defined by ``Gridworld``. The ``model`` method enumerates all three
    possible outcomes, including duplicate next states when movement is blocked.

    Args:
        layout_string: Rectangular ASCII representation of the grid.
        action_labels: Labels for the four actions, ordered as up, right, down,
            and left.
        goal_reward: Reward for a transition into a goal cell.
        step_reward: Reward for any other transition.
        tabular: If true, observations are integer state IDs. If false, they are
            raw ``(x, y)`` coordinates.
        render_mode: ``None`` to disable rendering, ``"human"`` for a window, or
            ``"rgb_array"`` for an RGB image returned by ``render``.

    Note:
        Rendering requires the optional ``render`` dependency extra.
    """

    def _sample_random_elements(self, state, action):
        return [self._noisy_action(action)]

    def _next_state(self, state, action, noisy_action):
        next_state, _ = super()._next_state(state, noisy_action)
        if action == noisy_action:
            return next_state, 0.8
        return next_state, 0.1

    def _noisy_action(self, action):
        p = self.np_random.random()
        # 10% chance: rotate the action clockwise
        if 0.8 <= p < 0.9:
            action += 1
        # 10% chance: rotate the action counter-clockwise
        elif 0.9 <= p:
            action -= 1
        return action % self.action_space.n

    def _generate_transitions(self, state, action):
        for i in [-1, 0, 1]:
            noisy_action = (action + i) % self.action_space.n
            yield self._deterministic_step(state, action, noisy_action)
