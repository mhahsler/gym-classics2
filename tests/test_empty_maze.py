"""The empty maze has configurable dimensions and opposite-corner endpoints."""

import gymnasium as gym
import pytest

from gym_classics2.envs.gym_classics2.empty_maze import EmptyMazeGridworld


@pytest.mark.parametrize("tabular", [True, False])
@pytest.mark.parametrize("goal_reward, step_reward", [(1.0, 0.0), (5.0, -0.1)])
@pytest.mark.parametrize("dims", [None, (3, 5), (1, 4), (4, 1)])
def test_empty_maze_layout_and_path(tabular, goal_reward, step_reward, dims):
    options = {} if dims is None else {"dims": dims}
    width, height = (4, 4) if dims is None else dims
    with gym.make(
        "EmptyMaze-v0", tabular=tabular,
        goal_reward=goal_reward, step_reward=step_reward, **options,
    ) as gym_env:
        env = gym_env.unwrapped
        assert isinstance(env, EmptyMazeGridworld)
        assert env.dims == (width, height)
        assert env.start_states == ((0, 0),)
        assert env.goal_states == ((width - 1, height - 1),)
        assert len(env.states()) == width * height
        assert all(env.is_reachable((x, y)) for x in range(width) for y in range(height))

        state, _ = gym_env.reset(seed=42)
        assert gym_env.observation_space.contains(state)
        actions = [env.action2id("up")] * (height - 1) + [env.action2id("right")] * (width - 1)
        for index, action in enumerate(actions):
            next_states, rewards, terminals, probabilities = env.model(state, action)
            next_state, reward, terminated, truncated, _ = gym_env.step(action)
            assert gym_env.observation_space.contains(next_state)
            assert next_states == [next_state]
            assert rewards[0] == reward
            assert terminals[0] == terminated
            assert probabilities.tolist() == [1.0]
            assert not truncated
            assert terminated == (index == len(actions) - 1)
            assert reward == (goal_reward if terminated else step_reward)
            state = next_state

        raw_state = env.id2state(state) if tabular else state
        assert raw_state == (width - 1, height - 1)


@pytest.mark.parametrize("dims", [(0, 4), (4, -1), (1, 1), (2.5, 4)])
def test_empty_maze_rejects_invalid_dimensions(dims):
    with pytest.raises(ValueError, match="dims"):
        EmptyMazeGridworld(dims=dims)
