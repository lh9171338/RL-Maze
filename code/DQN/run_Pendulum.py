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
    env = gym.make('Pendulum-v0')
    env = env.unwrapped

    logging.info(env.action_space)
    logging.info(env.observation_space)

    action_space = 11
    planner = DQNPlanner(num_action=action_space,
                         num_observation=env.observation_space.shape[0],
                         hidden_dims=20,
                         learning_rate=0.01,
                         epsilon=1.0,
                         epsilon_decay=lambda x: max(0.1, x - 0.0005),
                         replace_target_iter=200,
                         memory_size=3000,
                         )

    obs = env.reset()
    env.render()
    for step in range(20000):
        action = planner.choose_action(obs)
        f_action = 2 * (action - (action_space - 1) / 2)     # convert to [-2 ~ 2] float actions

        obs_, reward, done, _ = env.step(np.array([f_action]))
        reward /= 10
        env.render()

        planner.store_transition(obs=obs, action=action, reward=reward, obs_=obs_, done=False)
        obs = obs_

        if step >= planner.memory_size:
            planner.learn()
            planner.update()
        logging.info(f'step: {step} reward: {reward} epsilon: {planner.epsilon:.3f}')

    planner.plot_loss()
