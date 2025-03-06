import sys

sys.path.append('..')
import gym
import tqdm
import matplotlib.pyplot as plt
import logging
import numpy as np
import torch
from planner import PolicyGradientPlanner


def set_seed(seed=1):
    torch.autograd.set_detect_anomaly(True)
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = gym.make('Pendulum-v0')
    env = env.unwrapped

    logging.info(env.action_space)
    logging.info(env.action_space.low)
    logging.info(env.action_space.high)
    logging.info(env.observation_space)

    set_seed()
    continuous = True
    action_space = 11
    if continuous:
        planner = PolicyGradientPlanner(action_bound=(env.action_space.low[0], env.action_space.high[0]),
                                        num_observation=env.observation_space.shape[0],
                                        hidden_dims=30,
                                        continuous=continuous,
                                        learning_rate=0.001,
                                        reward_decay=0.9,
                                        gpu=-1,
                                        )
    else:
        planner = PolicyGradientPlanner(num_action=action_space,
                                        num_observation=env.observation_space.shape[0],
                                        hidden_dims=30,
                                        continuous=continuous,
                                        learning_rate=0.01,
                                        reward_decay=0.9,
                                        gpu=-1,
                                        )

    reward_list = []
    momentum = 0.99
    render = False
    for episode in tqdm.trange(1000):
        render = episode >= 990
        obs = env.reset()
        if render:
            env.render()

        step = 0
        while True:
            step += 1

            action = planner.choose_action(obs)
            if continuous:
                obs_, reward, done, _ = env.step(np.array([action]))
            else:
                f_action = 2 * (action - (action_space - 1) / 2)
                obs_, reward, done, _ = env.step(np.array([f_action]))
            if render:
                env.render()
            reward /= 10

            planner.store_transition(obs=obs, action=action, reward=reward)
            obs = obs_

            if step >= 200:
                ep_rs_sum = sum(planner.reward_list)
                if not reward_list:
                    reward_list.append(ep_rs_sum)
                else:
                    reward_list.append(reward_list[-1] * momentum + ep_rs_sum * (1 - momentum))

                planner.learn()

                logging.info(f'running_reward: {reward_list[-1]:.2f} step: {step}')
                break

    plt.figure()
    plt.plot(planner.vt)
    plt.xlabel('episode steps')
    plt.ylabel('normalized state-action value')

    plt.figure()
    plt.plot(reward_list)
    plt.xlabel('episode steps')
    plt.ylabel('episode reward')
    plt.show()
