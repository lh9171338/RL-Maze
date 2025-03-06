import sys
sys.path.append('..')
import sys
import threading
import logging
import numpy as np
from PyQt5.QtWidgets import QApplication
import environment
from planner import DQNPlanner


def thread_job(env: any):
    planner = DQNPlanner(num_action=env.num_action,
                         num_observation=2,
                         hidden_dims=[128, 64],
                         learning_rate=0.001,
                         replace_target_iter=100,
                         memory_size=2000,
                         batch_size=32,
                         epsilon=1.0,
                         epsilon_decay=lambda x: max(0.01, x * 0.995)
                         )

    # Train
    reward_list = []
    for episode in range(1000):
        obs = env.reset()
        obs = obs[:2]
        env.render()

        step = 0
        while True:
            step += 1

            action = planner.choose_action(obs)
            obs_, reward, done = env.step(action)
            obs_ = obs_[:2]
            env.render()
            planner.store_transition(obs=obs, action=action, reward=reward, obs_=obs_, done=done)
            obs = obs_

            if planner.memory_counter >= planner.batch_size:
                planner.learn()

            if done:
                reward_list.append(reward > 0)
                logging.info(f'episode: {episode} reward: {reward} step: {step} epsilon: {planner.epsilon:.2f}')
                break
        planner.update()

    planner.plot_loss()
    print(f'Train mean reward = {np.mean(reward_list)}')

    # Test
    obs = env.reset()
    obs = obs[:2]
    env.render()
    step = 0

    while True:
        step += 1
        action = planner.choose_best_action(obs)
        obs_, reward, done = env.step(action)
        obs_ = obs_[:2]
        env.render()
        planner.store_transition(obs=obs, action=action, reward=reward, obs_=obs_, done=done)
        planner.learn()
        obs = obs_

        if done:
            logging.info(f'reward: {reward} step: {step} epsilon: {planner.epsilon:.2f}')
            break


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)
    env = environment.make('MazeEnvironment', map_file='../../map/map-05.png')
    if env.init():
        env.reset()
        env.render()

        thread = threading.Thread(target=thread_job, args=(env,))
        thread.start()
        sys.exit(app.exec_())
