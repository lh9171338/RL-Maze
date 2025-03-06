import numpy as np
from abc import ABC
import pandas as pd
import heapq
import logging
from typing import Union
from .base_planner import Planner


class Node:
    def __init__(self, position: np.ndarray, parent: any = None, child: any = None, G: int = 0, F: int = 0):
        self.position = position
        self.parent = parent
        self.child = child
        self.G = G
        self.F = F

    def __lt__(self, other: any) -> bool:
        if self.F == other.F:
            return self.G < other.G
        else:
            return self.F < other.F

    def __str__(self) -> str:
        return f'position: {self.position} parent: {self.parent} child: {self.child} G: {self.G} F: {self.F}'


class PathPlanner(Planner, ABC):
    def __init__(self, movements, **kwargs):
        super().__init__(**kwargs)

        self.table = pd.DataFrame(columns=['node', 'status'], dtype=object)
        self.priority_queue = []
        self.state = None
        self.movements = movements

    def choose_action(self, obs: np.ndarray, **kwargs) -> Union[np.ndarray, None]:
        state = str(obs)
        if not self._check_state_exist(state):
            self.table.loc[state, 'node'] = Node(position=obs)
            return obs
        else:
            node = self.table.loc[state, 'node']
            if node.child is not None:
                return node.child.position
            else:
                if self.state is None:
                    node = heapq.heappop(self.priority_queue)
                    self.state = str(node.position)
                    self.table.loc[self.state, 'status'] = 2  # in closed list

                while True:
                    node = self.table.loc[self.state, 'node']
                    position = node.position
                    for movement in self.movements:
                        position_ = position + movement
                        state_ = str(position_)
                        G = node.G + 1
                        if not self._check_state_exist(state_):
                            self.table.loc[state_, 'node'] = Node(position=position_, parent=node, G=G)
                            return position_
                        else:
                            data = self.table.loc[state_]
                            node_, status = data['node'], data['status']
                            if status == 1 and G < node_.G:
                                node_.parent = node
                                node_.G = G
                                self.table.loc[state, 'node'] = node_
                                return position_

                    if len(self.priority_queue):
                        node = heapq.heappop(self.priority_queue)
                        self.state = str(node.position)
                        self.table.loc[self.state, 'status'] = 2  # in closed list
                    else:
                        break

        logging.error('Fail to choose action')
        return None

    def _check_state_exist(self, state: str) -> bool:
        exist = True
        if state not in self.table.index:
            self.table = self.table.append(pd.DataFrame([[None, 0]], columns=['node', 'status'],
                                                        index=[state, ], dtype=object))
            exist = False
        return exist


class AstarPlanner(PathPlanner):
    def learn(self, obs_: np.ndarray, distance: int, done: bool, **kwargs) -> None:
        state_ = str(obs_)
        node, status = self.table.loc[state_, :]
        if status == 0:
            node.F = node.G + distance
            heapq.heappush(self.priority_queue, node)
            self.table.loc[state_, 'status'] = 1  # in open list

        if done:
            # Arrive at the target position
            while node.parent is not None:
                node.parent.child = node
                node = node.parent


class DijkstraPlanner(PathPlanner, ABC):
    def learn(self, obs_: np.ndarray, done: bool, **kwargs) -> None:
        state_ = str(obs_)
        node, status = self.table.loc[state_, :]
        if status == 0:
            heapq.heappush(self.priority_queue, node)
            self.table.loc[state_, 'status'] = 1  # in open list

        if done:
            # Arrive at the target position
            while node.parent is not None:
                node.parent.child = node
                node = node.parent
