from .base_environment import Environment
from .maze_environment import MazeEnvironment


__all__ = ['MazeEnvironment']


def make(name: str, **kwargs) -> Environment:
    env_dict = {'MazeEnvironment': MazeEnvironment,
                }
    assert name in env_dict, 'Unrecognized environment'
    env = env_dict[name](**kwargs)

    return env
