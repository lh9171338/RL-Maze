import sys
sys.path.append('..')
import gym
import numpy as np
import cv2
import logging
import torch
from planner import DQNPlanner


def set_seed(seed=1):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def render(env: any) -> None:
    block_size = 50
    row, col = env.s // env.ncol, env.s % env.ncol
    desc = env.desc
    rows, cols = desc.shape
    height, width = rows * block_size, cols * block_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            color = [255, 255, 255]
            if desc[i, j] == b'H':
                color = [255, 0, 0]
            elif desc[i, j] == b'G':
                color = [0, 255, 0]
            pt1 = [j * block_size, i * block_size]
            pt2 = [(j + 1) * block_size, (i + 1) * block_size]
            cv2.rectangle(image, pt1, pt2, color, -1)
            cv2.rectangle(image, pt1, pt2, [128, 128, 128], 5)
    pt1 = [col * block_size, row * block_size]
    pt2 = [(col + 1) * block_size, (row + 1) * block_size]
    cv2.rectangle(image, pt1, pt2, [0, 0, 255], -1)
    cv2.rectangle(image, pt1, pt2, [128, 128, 128], 5)

    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', image)
    cv2.waitKey(100)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=False)
    env = env.unwrapped

    logging.info(env.action_space)
    logging.info(env.observation_space)

    set_seed()
    planner = DQNPlanner(num_action=env.action_space.n,
                         num_observation=2,
                         hidden_dims=[128, 64],
                         learning_rate=0.001,
                         replace_target_iter=1,
                         memory_size=2000,
                         batch_size=32,
                         epsilon=1.0,
                         epsilon_decay=lambda x: max(0.01, x * 0.995)
                         )

    obs = env.reset()
    render(env)

    # Train
    total_step = 0
    reward_list = []
    for episode in range(1000):
        obs = env.reset()
        obs = np.array([obs % env.ncol, obs // env.ncol])
        step = 0
        while True:
            step += 1
            action = planner.choose_action(obs)
            obs_, reward, done, _ = env.step(action)
            obs_ = np.array([obs_ % env.ncol, obs_ // env.ncol])

            planner.store_transition(obs=obs, action=action, reward=reward, obs_=obs_, done=done)
            obs = obs_

            if total_step >= planner.batch_size:
                planner.learn()

            if done:
                reward_list.append(reward == 1)
                logging.info(f'episode: {episode} reward: {reward} step: {step} epsilon: {planner.epsilon:.2f}')
                break

            total_step += 1
        planner.update()

    planner.plot_loss()
    print(f'Train mean reward = {np.mean(reward_list)}')

    # Test
    obs = env.reset()
    obs = np.array([obs % env.ncol, obs // env.ncol])
    render(env)
    step = 0
    while True:
        step += 1
        action = planner.choose_best_action(obs)
        obs_, reward, done, _ = env.step(action)
        obs_ = np.array([obs_ % env.ncol, obs_ // env.ncol])
        render(env)
        obs = obs_

        if done:
            reward_list.append(reward)
            logging.info(f'reward: {reward} step: {step} epsilon: {planner.epsilon:.2f}')
            break
