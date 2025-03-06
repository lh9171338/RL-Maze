import sys
sys.path.append('..')
import gym
import logging
import numpy as np
import torch
from planner import DQNPlanner


def set_seed(seed=1):
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
    planner = DQNPlanner(num_action=env.action_space.n,
                         num_observation=env.observation_space.shape[0],
                         hidden_dims=10,
                         learning_rate=0.01,
                         epsilon=1.0,
                         epsilon_decay=lambda x: max(0.1, x - 0.001),
                         replace_target_iter=100,
                         memory_size=2000,
                         )

    total_step = 0
    for episode in range(100):
        obs = env.reset()
        env.render()
        episode_reward = 0
        step = 0
        while True:
            step += 1
            action = planner.choose_action(obs)
            obs_, _, done, _ = env.step(action)
            env.render()

            # The smaller theta and closer to center the better
            x, vx, theta, vtheta = obs_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2

            planner.store_transition(obs=obs, action=action, reward=reward, obs_=obs_, done=False)
            obs = obs_
            episode_reward += reward
            if total_step >= 1000:
                planner.learn()
                planner.update()

            if done:
                logging.info(f'episode: {episode} episode_reward: {episode_reward:.2f} step: {step} '
                             f'epsilon: {planner.epsilon:.2f}')
                break

            total_step += 1

    planner.plot_loss()
