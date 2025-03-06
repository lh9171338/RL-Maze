import sys
sys.path.append('..')
import gym
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from planner import DQNPlanner


def set_seed(seed=1):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(planner):
    obs = env.reset()
    # env.render()
    for step in range(20000):
        action = planner.choose_action(obs)
        # convert to [-2 ~ 2] float actions
        f_action = 2 * (action - (action_space - 1) / 2)

        obs_, reward, done, _ = env.step(np.array([f_action]))
        reward /= 10
        # env.render()

        planner.store_transition(obs=obs, action=action, reward=reward, obs_=obs_, done=False)
        obs = obs_

        if step >= planner.memory_size:
            planner.learn()
            planner.update()
        logging.info(f'step: {step} reward: {reward} epsilon: {planner.epsilon:.3f}')

    return planner.q_list


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = gym.make('Pendulum-v0')
    env = env.unwrapped

    logging.info(env.action_space)
    logging.info(env.observation_space)

    action_space = 11
    set_seed()
    natural_DQN = DQNPlanner(num_action=action_space,
                             num_observation=env.observation_space.shape[0],
                             hidden_dims=20,
                             learning_rate=0.005,
                             epsilon=1.0,
                             epsilon_decay=lambda x: max(0.1, x - 0.001),
                             replace_target_iter=200,
                             memory_size=3000,
                             )
    q_natural = train(natural_DQN)

    set_seed()
    double_DQN = DQNPlanner(num_action=action_space,
                            num_observation=env.observation_space.shape[0],
                            hidden_dims=20,
                            learning_rate=0.005,
                            epsilon=1.0,
                            epsilon_decay=lambda x: max(0.1, x - 0.001),
                            replace_target_iter=200,
                            memory_size=3000,
                            double=True
                            )
    q_double = train(double_DQN)

    plt.figure()
    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.savefig('../../figure/double-DQN.png', format='png', bbox_inches='tight')
    plt.show()
