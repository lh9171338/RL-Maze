import numpy as np
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.3)
        nn.init.constant_(m.bias, 0.1)


class Planner:
    def __repr__(self) -> str :
        return self.__class__.__name__

    def __init__(self, **kwargs):
        pass

    def choose_action(self, **kwargs) -> any:
        raise NotImplementedError

    def learn(self, **kwargs) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def update(self) -> None:
        pass

    def save(self, filename: str) -> None:
        pass

    def load(self, filename: str) -> None:
        pass
