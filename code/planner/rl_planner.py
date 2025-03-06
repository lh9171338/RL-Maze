from abc import ABC
import pandas as pd
import numpy as np
from typing import List
from .base_planner import Planner


class RLPlanner(Planner, ABC):
    def __init__(self, actions: List[int], learning_rate: float = 0.5, reward_decay: float = 0.9,
                 epsilon: float = 0.9, epsilon_decay: any = None, **kwargs):
        super().__init__(**kwargs)

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.actions = actions
        self.q_table = pd.DataFrame(columns=self.actions, dtype=float)
        self.p_table = pd.DataFrame(columns=self.actions, dtype=bool)

    def choose_action(self, obs: np.ndarray, greedy: bool = False, **kwargs) -> int:
        state = str(obs)
        self._check_state_exist(state)
        if np.random.uniform() >= self.epsilon or greedy:
            action = self._choose_best_action(obs)
        else:
            candidates = [action for action in self.actions if not self.p_table.loc[state, action]]
            action = np.random.choice(candidates)
        action = int(action)
        return action

    def _choose_best_action(self, obs: np.ndarray) -> int:
        state = str(obs)
        self._check_state_exist(state)
        state_action = self.q_table.loc[state, :]
        action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        return action

    def _check_state_exist(self, state: str) -> None:
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns, name=state, dtype=float))
            self.p_table = self.p_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns, name=state, dtype=bool))

    def save(self, filename: None) -> None:
        self.q_table.to_csv(filename)

    def load(self, filename: str) -> None:
        self.q_table = pd.read_csv(filename, index_col=0)

    def update(self) -> None:
        if self.epsilon_decay is not None:
            self.epsilon = self.epsilon_decay(self.epsilon)


class QLearningPlanner(RLPlanner):
    def learn(self, obs: np.ndarray, action: int, obs_: np.ndarray, reward: int, done: bool, **kwargs) -> None:
        state = str(obs)
        state_ = str(obs_)
        self._check_state_exist(state_)
        q_predict = self.q_table.loc[state, action]
        if done:
            q_target = reward
            if reward < 0:
                self.p_table.loc[state, action] = True
        else:
            q_target = reward + self.gamma * self.q_table.loc[state_, :].max()
        # Update Q q_table
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)


class SarsaPlanner(RLPlanner):
    def learn(self, obs: np.ndarray, action: int, obs_: np.ndarray, action_: int, reward: int, done: bool, **kwargs):
        state = str(obs)
        state_ = str(obs_)
        self._check_state_exist(state_)
        q_predict = self.q_table.loc[state, action]
        if done:
            q_target = reward
            if reward < 0:
                self.p_table.loc[state, action] = True
        else:
            q_target = reward + self.gamma * self.q_table.loc[state_, action_]
        # Update Q q_table
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)


class SarsaLambdaPlanner(SarsaPlanner):
    def __init__(self, trace_decay: float = 0.9, **kwargs):
        super().__init__(**kwargs)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def learn(self, obs: np.ndarray, action: int, obs_: np.ndarray, action_: int, reward: int, done: bool, **kwargs):
        state = str(obs)
        state_ = str(obs_)
        self._check_state_exist(state_)
        q_predict = self.q_table.loc[state, action]
        if done:
            q_target = reward
            if reward < 0:
                self.p_table.loc[state, action] = True
        else:
            q_target = reward + self.gamma * self.q_table.loc[state_, action_]
        # Update Q q_table
        error = q_target - q_predict

        # self.eligibility_trace.loc[state, action] += 1
        self.eligibility_trace.loc[state, :] *= 0
        self.eligibility_trace.loc[state, action] = 1

        if error > 0:
            self.q_table += self.lr * error * self.eligibility_trace
        else:
            self.q_table.loc[state, action] += self.lr * error
        self.eligibility_trace *= self.gamma * self.lambda_

    def _check_state_exist(self, state: str) -> None:
        if state not in self.q_table.index:
            to_be_append = pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state, dtype=float)
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)
            self.p_table = self.p_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.p_table.columns, name=state, dtype=bool))

    def reset(self) -> None:
        self.eligibility_trace *= 0
