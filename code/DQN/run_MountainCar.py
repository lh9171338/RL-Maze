import sys
sys.path.append('..')
import gym
import tqdm
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
    env = gym.make('MountainCar-v0')
    env = env.unwrapped

    logging.info(env.action_space)
    logging.info(env.observation_space)

    set_seed()
    planner = DQNPlanner(num_action=env.action_space.n,
                         num_observation=env.observation_space.shape[0],
                         hidden_dims=20,
                         learning_rate=0.005,
                         replace_target_iter=100,
                         memory_size=3000,
                         epsilon=1.0,
                         epsilon_decay=lambda x: max(0.1, x - 0.0002)
                         )

    total_step = 0
    for episode in tqdm.trange(10):
        obs = env.reset()
        env.render()

        step = 0
        while True:
            step += 1

            action = planner.choose_action(obs)
            obs_, _, done, _ = env.step(action)
            env.render()

            position, velocity = obs_
            reward = abs(position - (-0.5))
            planner.store_transition(obs=obs, action=action, reward=reward, obs_=obs_, done=done)
            obs = obs_
            if total_step >= 1000:
                planner.learn()
                planner.update()

            if done:
                logging.info(f'episode_reward: {reward:.2f} step: {step} epsilon: {planner.epsilon:.2f}')
                break

            total_step += 1

    planner.plot_loss()
