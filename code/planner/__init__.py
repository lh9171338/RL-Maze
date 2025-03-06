from .base_planner import Planner
from .path_planner import AstarPlanner, DijkstraPlanner
from .rl_planner import QLearningPlanner, SarsaPlanner, SarsaLambdaPlanner
from .dqn_planner import DQNPlanner, PrioritizedReplayDQNPlanner
from .pg_planner import PolicyGradientPlanner
from .ac_planner import ActorCriticPlanner


__all__ = ['AstarPlanner', 'DijkstraPlanner', 'QLearningPlanner', 'SarsaPlanner', 'SarsaLambdaPlanner', 'DQNPlanner',
           'PrioritizedReplayDQNPlanner', 'PolicyGradientPlanner', 'ActorCriticPlanner']


def make(name: str, **kwargs) -> Planner:
    planner_dict = {'AstarPlanner': AstarPlanner,
                    'DijkstraPlanner': DijkstraPlanner,
                    'QLearningPlanner': QLearningPlanner,
                    'SarsaPlanner': SarsaPlanner,
                    'SarsaLambdaPlanner': SarsaLambdaPlanner,
                    'DQNPlanner': DQNPlanner,
                    'PrioritizedReplayDQNPlanner': PrioritizedReplayDQNPlanner,
                    'PolicyGradientPlanner': PolicyGradientPlanner,
                    'ActorCriticPlanner': ActorCriticPlanner
                    }
    assert name in planner_dict, 'Unrecognized planner'
    planner = planner_dict[name](**kwargs)

    return planner
