import sys

sys.path.append('..')
import gym
import matplotlib.pyplot as plt
import logging
import numpy as np
import torch
from planner import ActorCriticPlanner


def set_seed(seed=2):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = gym.make('CartPole-v0')
    env = env.unwrapped

    logging.info(env.action_space)
    logging.info(env.observation_space)

    set_seed()
    planner = ActorCriticPlanner(
        num_action=env.action_space.n,
        num_observation=env.observation_space.shape[0],
        continuous=False,
        hidden_dims=20,
        learning_rate1=0.001,
        learning_rate2=0.01,
        gpu=-1
    )

    reward_list = []
    momentum = 0.99
    render = False
    for episode in range(1000):
        obs = env.reset()
        if render:
            env.render()
        episode_reward = 0
        step = 0
        while True:
            step += 1
            action = planner.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            if render:
                env.render()

            if done:
                reward = -20

            planner.learn(obs=obs, action=action, reward=reward, obs_=obs_)
            obs = obs_
            episode_reward += reward
            if done or step >= 1000:
                if not reward_list:
                    reward_list.append(episode_reward)
                else:
                    reward_list.append(reward_list[-1] * momentum + episode_reward * (1 - momentum))
                if reward_list[-1] > 200:
                    render = True
                logging.info(f'episode: {episode} running_reward: {reward_list[-1]:.2f} step: {step}')
                break

    plt.figure()
    plt.plot(reward_list)
    plt.xlabel('episode steps')
    plt.ylabel('episode reward')
    plt.show()
