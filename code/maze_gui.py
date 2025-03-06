import os
import sys
import logging
import argparse
import threading
import tqdm
from typing import List, Union
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from environment import MazeEnvironment
import planner as pl


class Maze(QMainWindow):
    def __init__(self, planners: List[str], model_path: str = None):
        super().__init__()

        # Variables
        self.initialized = False
        self.running = False
        self.planners = planners
        self.env = None
        self.planner = None
        self.thread = None
        self.model_path = model_path

        # UI
        self.load_icon = QIcon('../icon/load.png')
        self.run_icon = QIcon('../icon/run.png')
        self.running_icon = QIcon('../icon/running.png')

        self.tool_bar = QToolBar()
        self.tool_load = self.tool_bar.addAction(self.load_icon, 'Load')
        self.tool_bar.addSeparator()
        self.tool_run = self.tool_bar.addAction('Run')
        self.addToolBar(self.tool_bar)

        self.label_image = QLabel()
        self.layout_image = QVBoxLayout()
        self.layout_image.addWidget(self.label_image, alignment=Qt.AlignCenter)
        self.layout_planner = QVBoxLayout()

        self.list_planner = QListWidget()
        self.list_planner.addItems(self.planners)
        self.list_planner.setCurrentRow(0)
        self.layout_planner.addStretch(1)
        self.layout_planner.addWidget(self.list_planner, alignment=Qt.AlignCenter)
        self.layout_planner.addStretch(1)

        self.layout = QHBoxLayout()
        self.layout.addStretch(1)
        self.layout.addLayout(self.layout_image)
        self.layout.addStretch(1)
        self.layout.addLayout(self.layout_planner)
        self.layout.addStretch(1)
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        self.setWindowTitle(self.__str__())

        # Register callback
        self.tool_load.triggered.connect(self.tool_load_callback)
        self.tool_run.triggered.connect(self.tool_run_callback)
        self.list_planner.clicked.connect(self.list_planner_callback)
        self.tool_load.setShortcut('Ctrl+O')
        self.tool_run.setShortcut('Ctrl+R')

        self.widget_update()

        self.show()

    def widget_update(self):
        if not self.initialized:
            self.tool_load.setEnabled(True)
            self.tool_run.setIcon(self.run_icon)
            self.tool_run.setEnabled(False)
            self.list_planner.setEnabled(False)
        else:
            self.tool_load.setEnabled(not self.running)
            self.tool_run.setIcon(self.running_icon if self.running else self.run_icon)
            self.tool_run.setEnabled(not self.running)
            self.list_planner.setEnabled(not self.running)

    def tool_load_callback(self, map_file: Union[str, None] = None):
        if not map_file:
            map_file = \
                QFileDialog.getOpenFileName(caption='Open map file', directory='../map',
                                            filter='Image Files(*.jpg *.png);;All Files(*)')[0]
        if not os.path.isfile(map_file):
            return
        env = MazeEnvironment(map_file, self.label_image)
        if env.init():
            # Initialize environment
            self.env = env
            self.env.reset()
            self.env.render()

            # Initialize planner
            self.list_planner_callback()

            # Update widget
            self.initialized = True
            self.widget_update()

    def tool_run_callback(self):
        self.running = True
        self.widget_update()

        self.thread = threading.Thread(target=self.thread_job)
        self.thread.start()

    def list_planner_callback(self):
        index = self.list_planner.currentRow()
        name = self.planners[index]
        self.planner = pl.make(name,
                               actions=self.env.actions,
                               movements=self.env.movements,
                               epsilon_min=0.8,
                               epsilon_increment=0.01,
                               )

    def thread_job(self):
        planner_name = str(self.planner)
        if planner_name in ['AstarPlanner', 'DijkstraPlanner']:
            for _ in tqdm.trange(2, desc=planner_name):
                # Initial observation
                obs = self.env.reset()
                obs = obs[:2]
                self.env.render()

                step = 0
                while True:
                    step += 1

                    # Planner choose action based on observation
                    action = self.planner.choose_action(obs)
                    logging.debug(f'Choose: {action}')
                    if action is None:
                        logging.info(f'Fail, take {step} steps')
                        break

                    # Planner take action and get child observation and reward
                    obs_, reward, done = self.env.step(action)
                    obs_, distance = obs_[:2], obs_[2]
                    logging.debug(f'Observe: {obs_}, {distance}, {reward}')

                    # Update environment
                    self.env.render()

                    # Planner learn from this transition
                    self.planner.learn(obs_=obs_, distance=distance, done=reward > 0)
                    obs = obs_

                    # Break while loop when end of this epoch
                    if reward > 0:
                        logging.info(f'Finish, take {step} steps')
                        break

        elif planner_name in ['QLearningPlanner']:
            for _ in tqdm.trange(50, desc=planner_name):
                # Initial observation
                obs = self.env.reset()
                obs = obs[:2]
                self.env.render()

                step = 0
                while True:
                    step += 1

                    # Planner choose action based on observation
                    action = self.planner.choose_action(obs)
                    logging.debug(f'Choose: {action}')

                    # Planner take action and get child observation and reward
                    obs_, reward, done = self.env.step(action)
                    obs_, distance = obs_[:2], obs_[2]
                    logging.debug(f'Observe: {obs_}, {distance}, {reward}')

                    # Update environment
                    self.env.render()

                    # Planner learn from this transition
                    self.planner.learn(obs=obs, action=action, obs_=obs_, reward=reward * 1e6 - 1, done=done)
                    obs = obs_

                    # Break while loop when end of this epoch
                    if reward > 0:
                        self.planner.update()
                        logging.info(f'steps: {step} epsilon: {self.planner.epsilon}')
                        break
        elif planner_name in ['SarsaPlanner', 'SarsaLambdaPlanner']:
            for _ in tqdm.trange(50, desc=planner_name):
                # Initial observation
                obs = self.env.reset()
                obs = obs[:2]
                self.env.render()
                self.planner.reset()

                # Planner choose action based on observation
                action = self.planner.choose_action(obs)
                logging.debug(f'Choose: {action}')

                step = 0
                while True:
                    step += 1

                    # Planner take action and get child observation and reward
                    obs_, reward, done = self.env.step(action)
                    obs_, distance = obs_[:2], obs_[2]
                    logging.debug(f'Observe: {obs_}, {distance}, {reward}')

                    # Update environment
                    self.env.render()

                    # Planner choose action based on observation
                    action_ = self.planner.choose_action(obs_)

                    # Planner learn from this transition
                    self.planner.learn(obs=obs, action=action, obs_=obs_, action_=action_, reward=reward * 1e6 - 1,
                                       done=done)
                    obs = obs_
                    action = action_

                    # Break while loop when end of this epoch
                    if reward > 0:
                        self.planner.update()
                        logging.info(f'steps: {step} epsilon: {self.planner.epsilon}')
                        break

        # Save model
        if self.model_path is not None:
            dir, filename = os.path.split(self.env.map_file)
            model_file = os.path.join(self.model_path, f'{planner_name}-{os.path.splitext(filename)[0]}.csv')
            self.planner.save(model_file)

        # Update widget
        self.running = False
        self.widget_update()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--map_file', type=str, default='../map/map-01.png', help='map file')
    parser.add_argument('-p', '--model_path', type=str, default='../model', help='model path')
    opts = parser.parse_args()
    print(opts)

    map_file = opts.map_file
    model_path = opts.model_path
    os.makedirs(model_path, exist_ok=True)
    planners = ['SarsaLambdaPlanner', 'SarsaPlanner', 'QLearningPlanner', 'AstarPlanner',
                'DijkstraPlanner']

    maze = Maze(planners, model_path)
    maze.tool_load_callback(map_file)
    sys.exit(app.exec_())
