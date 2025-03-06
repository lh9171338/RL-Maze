import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List
from .base_planner import Planner, init_weights


class DQN(nn.Module):
    def __init__(self, num_observation: int, num_action: int, hidden_dims: Union[List[int], int], act: any = nn.ReLU):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        fcs = []
        for i in range(len(hidden_dims)):
            in_dim = num_observation if i == 0 else hidden_dims[i - 1]
            fcs.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dims[i]),
                    act()
                )
            )
        fcs.append(nn.Linear(hidden_dims[-1], num_action))
        self.fc = nn.Sequential(*fcs)

        self.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class DuelingDQN(nn.Module):
    def __init__(self, num_observation: int, num_action: int, hidden_dims: Union[List[int], int], act: any = nn.ReLU):
        super().__init__()

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        fcs = []
        for i in range(len(hidden_dims)):
            in_dim = num_observation if i == 0 else hidden_dims[i - 1]
            fcs.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dims[i]),
                    act()
                )
            )
        self.fc = nn.Sequential(*fcs)
        self.fc_a = nn.Linear(hidden_dims[-1], num_action)
        self.fc_v = nn.Linear(hidden_dims[-1], 1)

        self.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        a = self.fc_a(x)
        v = self.fc_v(x)
        out = v + (a - a.mean(dim=-1, keepdim=True))
        return out


class DQNPlanner(Planner):
    def __init__(self, num_action: int, num_observation: int, hidden_dims: Union[List[int], int],
                 double: bool = False, dueling: bool = False, learning_rate: float = 0.01, reward_decay: float = 0.9,
                 epsilon: float = 0.9, epsilon_decay: any = None, replace_target_iter: int = 1,
                 memory_size: int = 2000, batch_size: int = 32, gpu: int = 0, **kwargs):
        super().__init__(**kwargs)

        use_gpu = gpu >= 0 and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{gpu}' if use_gpu else 'cpu')
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.num_action = num_action
        self.num_observation = num_observation

        self.double = double    # Double DQN
        self.dueling = dueling  # Dueling DQN

        # initialize zero memory [obs, obs_, action, reward, done]
        self.memory = np.zeros((self.memory_size, self.num_observation * 2 + 3))
        self.memory_counter = 0

        net = DuelingDQN if self.dueling else DQN
        self.eval_net = net(self.num_observation, self.num_action, hidden_dims).to(self.device)
        self.target_net = net(self.num_observation, self.num_action, hidden_dims).to(self.device)
        self.learn_step_counter = 0

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)
        self.loss_list = []

        self.momentum = 0.99
        self.running_q = 0
        self.q_list = []
        self.accum_done_list = []

    def choose_action(self, obs: np.ndarray, greedy: bool = False, **kwargs) -> int:
        if np.random.uniform() >= self.epsilon or greedy:
            action = self._choose_best_action(obs)
        else:
            action = np.random.randint(self.num_action)
        return action

    def _choose_best_action(self, obs: np.ndarray) -> int:
        x = torch.FloatTensor(obs)[None].to(self.device)
        y = self.eval_net(x)[0]
        q, action = y.max(dim=0)
        q, action = q.item(), action.item()

        self.running_q = self.momentum * self.running_q + (1.0 - self.momentum) * q
        self.q_list.append(self.running_q)

        return action

    def store_transition(self, obs: np.ndarray, action: int, obs_: np.ndarray, reward: int, done: bool) -> None:
        index = self.memory_counter % self.memory_size
        transition = np.hstack((obs, obs_, [action, reward, done]))
        self.memory[index] = transition
        self.memory_counter += 1

    def learn(self, **kwargs) -> None:
        # Check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        batch_obs = torch.FloatTensor(batch_memory[:, :self.num_observation]).to(self.device)
        batch_obs_ = torch.FloatTensor(batch_memory[:, self.num_observation:2 * self.num_observation]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, -3:-2]).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, -2:-1]).to(self.device)
        batch_done = torch.FloatTensor(batch_memory[:, -1:]).to(self.device)

        # Change q_target w.r.t q_eval's action
        self.eval_net.train()
        self.target_net.eval()

        q_predict = self.eval_net(batch_obs).gather(1, batch_action)  # shape (batch, 1)
        q_next = self.target_net(batch_obs_)
        if self.double:
            q_next_action = self.eval_net(batch_obs_).argmax(dim=-1, keepdim=True)
            q_target = batch_reward + (1 - batch_done) * self.gamma * q_next.gather(1, q_next_action)
        else:
            q_target = batch_reward + (1 - batch_done) * self.gamma * q_next.max(1)[0].view(-1, 1)
        loss = F.mse_loss(q_predict, q_target)
        self.loss_list.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        accum_done = batch_memory[:, -1:].sum()
        if len(self.accum_done_list):
            accum_done += self.accum_done_list[-1]
        self.accum_done_list.append(accum_done)

    def plot_loss(self):
        plt.figure()
        plt.plot(np.arange(len(self.loss_list)), self.loss_list)
        plt.ylabel('Lost')
        plt.xlabel('training steps')
        plt.show()

    def save(self, filename: str) -> None:
        torch.save(self.eval_net.state_dict(), filename)

    def load(self, filename: str) -> None:
        state_dict = torch.load(filename, map_location=self.device)
        self.eval_net.load_state_dict(state_dict)

    def update(self):
        if self.epsilon_decay is not None:
            self.epsilon = self.epsilon_decay(self.epsilon)


class PrioritizedReplayDQNPlanner(DQNPlanner):
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001, eps: float = 1e-5,
                 **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.eps = eps

        # initialize priority
        self.priority = np.zeros(self.memory_size)

    def _batch_update(self, sample_index, abs_error):
        abs_error += self.eps
        p = np.power(abs_error, self.alpha)
        self.priority[sample_index] = p

    def _batch_sample(self):
        # Sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            p = self.priority / self.priority.sum()
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, p=p)
        else:
            p = self.priority[:self.memory_counter] / self.priority[:self.memory_counter].sum()
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, p=p)
        weight = np.power(p / p.min(), -self.beta)
        self.beta = min(1., self.beta + self.beta_increment)
        return sample_index, weight

    def store_transition(self, obs: np.ndarray, action: int, obs_: np.ndarray, reward: int, done: bool) -> None:
        index = self.memory_counter % self.memory_size
        transition = np.hstack((obs, obs_, [action, reward, done]))
        self.memory[index] = transition
        max_p = self.priority.max()
        if max_p == 0.0:
            max_p = 1.0
        self.priority[index] = max_p
        self.memory_counter += 1

    def learn(self, **kwargs) -> None:
        # Check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index, weight = self._batch_sample()
        batch_memory = self.memory[sample_index, :]
        batch_obs = torch.FloatTensor(batch_memory[:, :self.num_observation]).to(self.device)
        batch_obs_ = torch.FloatTensor(batch_memory[:, self.num_observation:2 * self.num_observation]).to(self.device)
        batch_action = torch.LongTensor(batch_memory[:, -3:-2]).to(self.device)
        batch_reward = torch.FloatTensor(batch_memory[:, -2:-1]).to(self.device)
        batch_done = torch.FloatTensor(batch_memory[:, -1:]).to(self.device)
        batch_weight = torch.FloatTensor(weight).to(self.device)

        # Change q_target w.r.t q_eval's action
        self.eval_net.train()
        self.target_net.eval()

        q_predict = self.eval_net(batch_obs).gather(1, batch_action)  # shape (batch, 1)
        q_next = self.target_net(batch_obs_)
        if self.double:
            q_next_action = self.eval_net(batch_obs_).argmax(dim=-1, keepdim=True)
            q_target = batch_reward + (1 - batch_done) * self.gamma * q_next.gather(1, q_next_action)
        else:
            q_target = batch_reward + (1 - batch_done) * self.gamma * q_next.max(1)[0].view(-1, 1)
        abs_error = (q_predict - q_target).abs().detach().cpu().numpy().squeeze()
        loss = (batch_weight * F.mse_loss(q_predict, q_target, reduction='none')).mean()
        self.loss_list.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._batch_update(sample_index, abs_error)

        accum_done = batch_memory[:, -1:].sum()
        if len(self.accum_done_list):
            accum_done += self.accum_done_list[-1]
        self.accum_done_list.append(accum_done)
