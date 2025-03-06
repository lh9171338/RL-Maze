import numpy as np
from PyQt5.QtWidgets import *


class Environment(QMainWindow):
    def __repr__(self) -> str :
        return self.__class__.__name__

    def __init__(self, sleep_duration: float = 0.0, render_frequency: int = 1, **kwargs):
        super().__init__()

        self.actions = None
        self.num_action = None
        self.num_observation = None
        self.sleep_duration = sleep_duration
        self.render_frequency = render_frequency
        self.render_counter = 0

    def init(self) -> bool:
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: any) -> any:
        raise NotImplementedError

    def render(self) -> None:
        raise NotImplementedError

    def set_render_frequency(self, render_frequency):
        self.render_frequency = render_frequency
