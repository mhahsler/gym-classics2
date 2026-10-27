_registry = (
    {
        'id': '5Walk-v0',
        'entry_point': 'gym_classics2.envs.gym_classics2.linear_walks:Walk5'
    },
    {
        'id': '19Walk-v0',
        'entry_point': 'gym_classics2.envs.gym_classics2.linear_walks:Walk19'
    },
    {
        'id': 'ClassicGridworld-v1',
        'entry_point': 'gym_classics2.envs.gym_classics2.classic_gridworld_v1:ClassicGridworld'
    },
    {
        'id': 'LMaze-v0',
        'entry_point': 'gym_classics2.envs.gym_classics2.L_maze:LMazeGridworld'
    },
    {
        'id': 'CliffWalk-v1',
        'entry_point': 'gym_classics2.envs.gym_classics2.cliff_walk_v1:CliffWalk'
    },
    {
        'id': 'DynaMaze-v0',
        'entry_point': 'gym_classics2.envs.gym_classics2.dyna_maze:DynaMaze',
    },
    {
        'id': 'FourRooms-v0',
        'entry_point': 'gym_classics2.envs.gym_classics2.four_rooms:FourRooms',
    },
    #{
    #    'id': 'JacksCarRental-v0',
    #    'entry_point': 'gym_classics2.envs.jacks_car_rental:JacksCarRental',
    #    'max_episode_steps': 100
    #},
    #{
    #    'id': 'JacksCarRentalModified-v0',
    #    'entry_point': 'gym_classics2.envs.jacks_car_rental:JacksCarRentalModified',
    #    'max_episode_steps': 100,
    #},
    # {
    #     'id': 'Racetrack1-v0',
    #     'entry_point': 'gym_classics2.envs.racetracks:Racetrack1',
    # },
    # {
    #     'id': 'Racetrack2-v0',
    #     'entry_point': 'gym_classics2.envs.racetracks:Racetrack2',
    # },
    {
        'id': 'SparseGridworld-v0',
        'entry_point': 'gym_classics2.envs.gym_classics2.sparse_gridworld:SparseGridworld',
    },
    {
        'id': 'WindyGridworld-v0',
        'entry_point': 'gym_classics2.envs.gym_classics2.windy_gridworld:WindyGridworld',
    }
)


_registered_backends = set()


def register(backend='gymnasium'):
    """Register all bundled environments with Gymnasium or legacy Gym.

    Importing ``gym_classics2`` automatically registers with Gymnasium.
    Explicit calls remain supported; repeated calls for a backend do nothing.

    Args:
        backend: Registry to use. ``"gymnasium"`` is the default and recommended
            value; ``"gym"`` supports legacy installations.

    Raises:
        AssertionError: If *backend* is neither ``"gymnasium"`` nor ``"gym"``.

    Example:
        >>> import gymnasium as gym
        >>> import gym_classics2
        >>> env = gym.make("ClassicGridworld-v1")
    """
    assert backend in {'gym', 'gymnasium'}
    if backend in _registered_backends:
        return

    if backend == 'gym':
        import gym
        register = gym.envs.register
    elif backend == 'gymnasium':
        import gymnasium
        register = gymnasium.register

    for kwargs in _registry:
        register(**kwargs)

    _registered_backends.add(backend)


register()
