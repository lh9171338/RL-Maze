import os
import cv2
import time
import numpy as np
from typing import Union, Tuple
from yacs.config import CfgNode
import logging
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from .base_environment import Environment


class MazeEnvironment(Environment):
    def __init__(self, map_file: str, qlabel: QLabel = None, interactive: bool = False, **kwargs):
        super().__init__(**kwargs)

        # Variables
        self.actions = np.array([0, 1, 2, 3])
        self.num_action = 4
        self.num_observation = 3
        self.movements = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        self.key_dict = {Qt.Key_Left: 0, Qt.Key_Right: 1, Qt.Key_Up: 2, Qt.Key_Down: 3}
        self.reward_dict = None

        self.map_file = map_file
        self.image = None
        self.original_image = None
        self.map = None
        self.start_position = None
        self.target_position = None
        self.block_size = None
        self.position = None
        self.last_position = None
        self.done = False

        # UI
        if interactive:
            self.keyReleaseEvent = self._key_callback

        if qlabel is None:
            # As QMainWindow
            self.label_image = QLabel()
            self.setCentralWidget(self.label_image)
            self.setWindowTitle(self.__str__())
            self.show()
        else:
            self.label_image = qlabel

    def _key_callback(self, event: QEvent) -> None:
        if self.done:
            return

        key = event.key()
        if key in self.key_dict:
            action = self.key_dict[key]
            logging.info(f'Action: {action}')

            observation, reward, done = self.step(action)
            self.render()
            self.done = reward == 1

    def init(self) -> bool:
        map_file = self.map_file
        if not os.path.isfile(map_file):
            logging.error(f'{map_file} does not exist')
            return False
        dir, filename = os.path.split(map_file)
        yaml_file = os.path.join(dir, os.path.splitext(filename)[0] + '.yaml')
        if not os.path.isfile(yaml_file):
            logging.error(f'{yaml_file} does not exist')
            return False

        image = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
        image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)[1]
        height, width = image.shape
        cfg = CfgNode.load_cfg(open(yaml_file))
        cfg.freeze()
        block_size = cfg.block_size
        rows = int(np.ceil(height / block_size))
        cols = int(np.ceil(width / block_size))
        map = np.zeros((rows, cols), bool)
        for i in range(rows):
            for j in range(cols):
                block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
                value = block.mean() > 128
                map[i, j] = value
                image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] = value * 255
        image = cv2.merge([image] * 3)

        start_position = np.asarray(cfg.start_position, dtype=np.int32)
        target_position = np.asarray(cfg.target_position, dtype=np.int32)
        self.start_position = start_position // block_size
        self.target_position = target_position // block_size
        self.block_size = block_size
        self.map = map
        self.image = image
        self.original_image = image
        return True

    def reset(self) -> np.ndarray:
        self.image = self.original_image.copy()
        self.position = self.start_position
        self.last_position = None
        self.render_counter = 0

        distance = np.abs(self.position - self.target_position).sum()
        observation = np.array([self.position[0], self.position[1], distance])

        return observation

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, int, bool]:
        if isinstance(action, int):
            movement = self.movements[action]
            position = self.position + movement
        else:
            position = np.asarray(action)

        x, y = position
        rows, cols = self.map.shape
        reward = -1
        done = True
        if 0 <= x < cols and 0 <= y < rows and self.map[y, x]:
            reward = 0
            done = False
            self.last_position = self.position
            self.position = position
        distance = np.abs(self.position - self.target_position).sum()
        if distance == 0:
            reward = 1
            done = True

        observation = np.array([self.position[0], self.position[1], distance])

        return observation, reward, done

    def render(self) -> None:
        image = self.image
        block_size = self.block_size
        target_position = self.target_position * block_size
        position = self.position * block_size

        if self.last_position is not None:
            last_position = self.last_position * block_size
            cv2.rectangle(image, last_position, last_position + block_size, [255, 0, 0], thickness=-1)
        cv2.rectangle(image, target_position, target_position + block_size, [0, 255, 0], thickness=-1)
        cv2.circle(image, position + block_size // 2, block_size // 2, [0, 0, 255], thickness=-1)

        if self.render_frequency > 0:
            if self.render_counter % self.render_frequency == 0:
                image = QImage(image.data, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format.Format_BGR888)
                pixmap = QPixmap(image).scaled(image.width(), image.height())
                self.label_image.setPixmap(pixmap)
            self.render_counter += 1
        time.sleep(self.sleep_duration)
