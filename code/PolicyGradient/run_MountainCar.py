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
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    logging.info(env.action_space)
    logging.info(env.observation_space)

    set_seed()
    planner = PolicyGradientPlanner(num_action=env.action_space.n,
                                    num_observation=env.observation_space.shape[0],
                                    hidden_dims=20,
                                    continuous=False,
                                    learning_rate=0.01,
                                    reward_decay=0.995,
                                    gpu=-1,
                                    )

    reward_list = []
    momentum = 0.99
    for episode in tqdm.trange(1000):
        render = episode > 990
        obs = env.reset()
        if render:
            env.render()

        step = 0
        while True:
            step += 1

            action = planner.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            if render:
                env.render()

            planner.store_transition(obs=obs, action=action, reward=reward)
            obs = obs_

            if done:
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
    plt.savefig('../../figure/policyGradient-vt.png', format='png', bbox_inches='tight')

    plt.figure()
    plt.plot(reward_list)
    plt.xlabel('episode steps')
    plt.ylabel('episode reward')
    plt.savefig('../../figure/policyGradient-reward.png', format='png', bbox_inches='tight')
    plt.show()
