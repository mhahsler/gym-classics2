import warnings


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


_backend = None

def register(backend='gymnasium'):
    global _backend
    if _backend is not None:
        warnings.warn("gym-classics environments were already registered for {}; "
                      "additional calls to `register()` are ignored.".format(_backend))
        return

    assert backend in {'gym', 'gymnasium'}
    _backend = backend

    if backend == 'gym':
        import gym
        register = gym.envs.register
    elif backend == 'gymnasium':
        import gymnasium
        register = gymnasium.register

    for kwargs in _registry:
        register(**kwargs)
