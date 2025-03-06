import sys
sys.path.append('..')
import gym
import tqdm
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
from planner import DQNPlanner, PrioritizedReplayDQNPlanner


def set_seed(seed=1):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(planner):
    step_list = []
    total_step = 0
    for episode in tqdm.trange(10):
        obs = env.reset()
        # env.render()

        step = 0
        while True:
            step += 1
            action = planner.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            # env.render()
            if done:
                reward = 10

            planner.store_transition(obs=obs, action=action, reward=reward, obs_=obs_, done=done)
            obs = obs_
            if total_step >= planner.memory_size:
                planner.learn()
                planner.update()

            if done:
                logging.info(f'episode_reward: {reward:.2f} step: {step} epsilon: {planner.epsilon:.2f}')
                step_list.append(total_step)
                break

            total_step += 1

    return np.asarray(step_list)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    logging.info(env.action_space)
    logging.info(env.observation_space)

    set_seed()
    natural_DQN = DQNPlanner(num_action=env.action_space.n,
                             num_observation=env.observation_space.shape[0],
                             hidden_dims=20,
                             learning_rate=0.005,
                             replace_target_iter=500,
                             memory_size=10000,
                             epsilon=1.0,
                             epsilon_decay=lambda x: max(0.1, x - 0.00005)
                             )
    step_natural = train(natural_DQN)

    set_seed()
    priority_DQN = PrioritizedReplayDQNPlanner(num_action=env.action_space.n,
                                               num_observation=env.observation_space.shape[0],
                                               hidden_dims=20,
                                               learning_rate=0.005,
                                               replace_target_iter=500,
                                               memory_size=10000,
                                               epsilon=1.0,
                                               epsilon_decay=lambda x: max(0.1, x - 0.00005)
                                               )

    step_priority = train(priority_DQN)

    done_natural = np.asarray(natural_DQN.accum_done_list)
    done_priority = np.asarray(priority_DQN.accum_done_list)

    # compare based on first success
    plt.figure()
    plt.plot(np.arange(len(step_natural)), step_natural - step_natural[0], c='b', label='natural DQN')
    plt.plot(np.arange(len(step_priority)), step_priority - step_priority[0], c='r',
             label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.savefig('../../figure/priority-DQN.png', format='png', bbox_inches='tight')

    plt.figure()
    plt.plot(np.arange(len(done_natural)), done_natural, c='b', label='natural DQN')
    plt.plot(np.arange(len(done_priority)), done_priority, c='r', label='DQN with prioritized replay')
    plt.legend(loc='best')
    plt.ylabel('total training time')
    plt.xlabel('episode')
    plt.grid()
    plt.show()
