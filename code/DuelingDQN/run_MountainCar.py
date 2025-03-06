import sys
sys.path.append('..')
import gym
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from planner import DQNPlanner


def set_seed(seed=25):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(planner):
    acc_r = [0]
    obs = env.reset()
    # env.render()
    for step in range(20000):
        action = planner.choose_action(obs)
        f_action = 2 * (action - (action_space - 1) / 2)  # convert to [-2 ~ 2] float actions

        obs_, reward, done, _ = env.step(np.array([f_action]))
        # env.render()
        reward /= 10
        acc_r.append(reward + acc_r[-1])

        planner.store_transition(obs=obs, action=action, reward=reward, obs_=obs_, done=False)
        obs = obs_

        if step >= planner.memory_size:
            planner.learn()
            planner.update()
        logging.info(f'step: {step} reward: {reward} epsilon: {planner.epsilon:.3f}')

    return acc_r


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = gym.make('Pendulum-v0')
    env = env.unwrapped

    logging.info(env.action_space)
    logging.info(env.observation_space)

    action_space = 25
    set_seed()
    natural_DQN = DQNPlanner(num_action=action_space,
                             num_observation=env.observation_space.shape[0],
                             hidden_dims=20,
                             learning_rate=0.001,
                             epsilon=1.0,
                             epsilon_decay=lambda x: max(0.1, x - 0.0005),
                             replace_target_iter=200,
                             memory_size=3000,
                             )
    r_natural = train(natural_DQN)

    set_seed()
    dueling_DQN = DQNPlanner(num_action=action_space,
                             num_observation=env.observation_space.shape[0],
                             hidden_dims=20,
                             learning_rate=0.001,
                             epsilon=1.0,
                             epsilon_decay=lambda x: max(0.1, x - 0.0005),
                             replace_target_iter=200,
                             memory_size=3000,
                             dueling=True,
                             )
    r_dueling = train(dueling_DQN)

    c_natural = natural_DQN.loss_list
    c_dueling = dueling_DQN.loss_list

    plt.figure()
    plt.plot(c_natural, c='r', label='natural')
    plt.plot(c_dueling, c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('cost')
    plt.xlabel('training steps')
    plt.grid()

    plt.figure()
    plt.plot(r_natural, c='r', label='natural')
    plt.plot(r_dueling, c='b', label='dueling')
    plt.legend(loc='best')
    plt.ylabel('accumulated reward')
    plt.xlabel('training steps')
    plt.grid()
    plt.savefig('../../figure/dueling-DQN.png', format='png', bbox_inches='tight')
    plt.show()
