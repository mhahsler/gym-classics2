"""Registration works on import and remains compatible with explicit calls."""

import subprocess
import sys
from types import SimpleNamespace
import warnings

import gymnasium as gym
import gym_classics2


def test_import_registers_environments():
    # A fresh interpreter ensures another test has not registered them first.
    subprocess.run(
        [sys.executable, "-c", """
import gymnasium as gym
assert 'ClassicGridworld-v1' not in gym.registry
import gym_classics2
for env_id in (
    '5Walk-v0', '19Walk-v0', 'ClassicGridworld-v1', 'LMaze-v0', 'EmptyMaze-v0',
    'CliffWalk-v1', 'DynaMaze-v0', 'FourRooms-v0',
    'SparseGridworld-v0', 'WindyGridworld-v0',
):
    with gym.make(env_id) as env:
        state, info = env.reset(seed=42)
        assert env.observation_space.contains(state)
        state, reward, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(state)
"""],
        check=True,
    )


def test_explicit_registration_is_silent_and_preserves_specs():
    spec = gym.spec("ClassicGridworld-v1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gym_classics2.register()
        gym_classics2.register(backend="gymnasium")
    assert gym.spec("ClassicGridworld-v1") is spec


def test_legacy_registry_can_be_registered_separately(monkeypatch):
    registrations = []
    legacy_gym = SimpleNamespace(
        envs=SimpleNamespace(register=lambda **kwargs: registrations.append(kwargs))
    )
    monkeypatch.setitem(sys.modules, "gym", legacy_gym)
    monkeypatch.setattr(gym_classics2, "_registered_backends", {"gymnasium"})

    gym_classics2.register(backend="gym")
    assert {entry["id"] for entry in registrations} == {
        entry["id"] for entry in gym_classics2._registry
    }
    count = len(registrations)
    gym_classics2.register(backend="gym")
    assert len(registrations) == count
