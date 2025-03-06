import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Tuple
from .base_planner import Planner, init_weights
from .pg_planner import DiscreteActorNet, ContinuousActorNet


class CriticNet(nn.Module):
    def __init__(self, num_observation: int, hidden_dims: Union[List[int], int], reward_decay: float = 0.9,
                 learning_rate: float = 0.01, act: any = nn.ReLU):
        super().__init__()

        self.gamma = reward_decay

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
        fcs.append(nn.Linear(hidden_dims[-1], 1))
        self.fc = nn.Sequential(*fcs)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

    def learn(self, obs, obs_, reward):
        self.train()
        with torch.no_grad():
            v_ = self(obs_)
        td_error = reward + self.gamma * v_ - self(obs)
        loss = (td_error ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return td_error


class ActorCriticPlanner(Planner):
    def __init__(self, continuous: bool, num_observation: int, hidden_dims: Union[List[int], int],
                 num_action: int = None, action_bound: Tuple[int, int] = None, learning_rate1: float = 0.01,
                 learning_rate2: float = 0.01, reward_decay: float = 0.9, gpu: int = 0, **kwargs):
        super().__init__(**kwargs)

        use_gpu = gpu >= 0 and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{gpu}' if use_gpu else 'cpu')
        self.continuous = continuous
        self.action_bound = action_bound

        if self.continuous:
            self.actor = ContinuousActorNet(
                num_observation=num_observation,
                action_bound=action_bound,
                hidden_dims=hidden_dims,
                learning_rate=learning_rate1
            )
        else:
            self.actor = DiscreteActorNet(
                num_observation=num_observation,
                num_action=num_action,
                hidden_dims=hidden_dims,
                learning_rate=learning_rate1
            )
        self.critic = CriticNet(
            num_observation=num_observation,
            hidden_dims=hidden_dims,
            reward_decay=reward_decay,
            learning_rate=learning_rate2
        )

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

    def choose_action(self, obs: np.ndarray, **kwargs) -> int:
        self.actor.eval()
        x = torch.FloatTensor(obs)[None].to(self.device)
        if self.continuous:
            normal_dist = self.actor(x)
            action = normal_dist.sample().item()
            action = np.clip(action, self.action_bound[0], self.action_bound[1])
        else:
            prob = self.actor(x)[0].detach().cpu().numpy()
            action = np.random.choice(range(prob.shape[0]), p=prob)
        return action

    def learn(self, obs: np.ndarray, action: float, obs_: np.ndarray, reward: float, **kwargs) -> None:
        obs = torch.FloatTensor(obs)[None].to(self.device)
        action = torch.FloatTensor([action])[None].to(self.device)
        obs_ = torch.FloatTensor(obs_)[None].to(self.device)
        reward = torch.FloatTensor([reward])[None].to(self.device)

        td_error = self.critic.learn(obs, obs_, reward).detach()
        self.actor.learn(obs, action, td_error)

    def save(self, filename: str) -> None:
        state_dict = {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}
        torch.save(state_dict, filename)

    def load(self, filename: str) -> None:
        state_dict = torch.load(filename, map_location=self.device)
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
