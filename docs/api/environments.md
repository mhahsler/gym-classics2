# Environment API

## Base environment

::: gym_classics2.envs.abstract.base_env.BaseEnv
    options:
      members:
        - start_states
        - states
        - state2id
        - id2state
        - is_reachable
        - actions
        - action2id
        - id2action
        - model

## Gridworld

::: gym_classics2.envs.abstract.gridworld.Gridworld
    options:
      members:
        - goal_states
        - reset
        - step
        - render
        - print
        - image
        - image_list

## Concrete environments

::: gym_classics2.envs.gym_classics2.classic_gridworld_v1.ClassicGridworld

::: gym_classics2.envs.gym_classics2.cliff_walk_v1.CliffWalk

::: gym_classics2.envs.gym_classics2.dyna_maze.DynaMaze

::: gym_classics2.envs.gym_classics2.four_rooms.FourRooms

::: gym_classics2.envs.gym_classics2.L_maze.LMazeGridworld

::: gym_classics2.envs.gym_classics2.sparse_gridworld.SparseGridworld

::: gym_classics2.envs.gym_classics2.windy_gridworld.WindyGridworld

::: gym_classics2.envs.gym_classics2.linear_walks.Walk5

::: gym_classics2.envs.gym_classics2.linear_walks.Walk19
