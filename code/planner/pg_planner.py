import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Tuple
from .base_planner import Planner, init_weights


def softplus(x, limit=30):
    mask = (x < limit).float()
    x = torch.log(1.0 + torch.exp(x * mask)) * mask + x * (1 - mask)
    return x


class ContinuousActorNet(nn.Module):
    def __init__(self, num_observation: int, action_bound: Tuple[int, int], hidden_dims: Union[List[int], int],
                 learning_rate: float = 0.01, act: any = nn.ReLU):
        super().__init__()

        self.action_bound = [(action_bound[1] + action_bound[0]) / 2, (action_bound[1] - action_bound[0]) / 2]

        # Build network
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
        self.mu = nn.Linear(hidden_dims[-1], 1)
        self.sigma = nn.Linear(hidden_dims[-1], 1)
        self.normal_dist = torch.distributions.Normal(0, 1)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        mu = self.mu(x).tanh()
        mu = mu * self.action_bound[1] + self.action_bound[0]
        sigma = softplus(self.sigma(x)) + 1e-4

        self.normal_dist = torch.distributions.Normal(mu, sigma)
        return self.normal_dist

    def learn(self, obs, action, td_error):
        self.train()

        self.forward(obs)
        prob = self.normal_dist.log_prob(action)
        exp_v = prob * td_error + 0.01 * self.normal_dist.entropy()
        loss = -exp_v.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DiscreteActorNet(nn.Module):
    def __init__(self, num_observation: int, num_action: int, hidden_dims: Union[List[int], int],
                 learning_rate: float = 0.01, act: any = nn.ReLU):
        super().__init__()

        # Build network
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
        self.out = nn.Sequential(
            nn.Linear(hidden_dims[-1], num_action),
            nn.Softmax(dim=-1)
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        self.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        x = self.out(x)
        return x

    def learn(self, obs, action, td_error):
        self.train()

        action = action.long()
        prob = self.forward(obs)
        log_prob = -prob.gather(dim=1, index=action).log()
        loss = (td_error * log_prob).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PolicyGradientPlanner(Planner):
    def __init__(self, continuous: bool, num_observation: int, hidden_dims: Union[List[int], int],
                 num_action: int = None, action_bound: Tuple[int, int] = None, learning_rate: float = 0.01,
                 reward_decay: float = 0.9, gpu: int = 0, **kwargs):
        super().__init__(**kwargs)

        use_gpu = gpu >= 0 and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{gpu}' if use_gpu else 'cpu')
        self.gamma = reward_decay
        self.continuous = continuous
        self.action_bound = action_bound

        self.obs_list, self.action_list, self.reward_list = [], [], []
        self.vt = None

        if self.continuous:
            self.net = ContinuousActorNet(
                num_observation=num_observation,
                action_bound=action_bound,
                hidden_dims=hidden_dims,
                learning_rate=learning_rate
            )
        else:
            self.net = DiscreteActorNet(
                num_observation=num_observation,
                num_action=num_action,
                hidden_dims=hidden_dims,
                learning_rate=learning_rate
            )
        self.net = self.net.to(self.device)

    def choose_action(self, obs: np.ndarray, **kwargs) -> int:
        self.net.eval()
        x = torch.FloatTensor(obs)[None].to(self.device)
        if self.continuous:
            normal_dist = self.net(x)
            action = normal_dist.sample().item()
            action = np.clip(action, self.action_bound[0], self.action_bound[1])
        else:
            prob = self.net(x)[0].detach().cpu().numpy()
            action = np.random.choice(range(prob.shape[0]), p=prob)
        return action

    def store_transition(self, obs: np.ndarray, action: float, reward: float) -> None:
        self.obs_list.append(obs)
        self.action_list.append(action)
        self.reward_list.append(reward)

    def learn(self, **kwargs) -> None:
        self._discount_and_norm_rewards()
        obs = torch.FloatTensor(self.obs_list).to(self.device)
        action = torch.FloatTensor(self.action_list).to(self.device)[:, None]
        vt = torch.FloatTensor(self.vt).to(self.device)[:, None]

        # Train on episode
        self.net.learn(obs, action, vt)

        self.obs_list, self.action_list, self.reward_list = [], [], []

    def save(self, filename: str) -> None:
        torch.save(self.net.state_dict(), filename)

    def load(self, filename: str) -> None:
        state_dict = torch.load(filename, map_location=self.device)
        self.net.load_state_dict(state_dict)

    def _discount_and_norm_rewards(self):
        # Discount episode rewards
        vt = np.zeros_like(self.reward_list)
        running_add = 0
        for t in reversed(range(0, len(self.reward_list))):
            running_add = running_add * self.gamma + self.reward_list[t]
            vt[t] = running_add

        # Normalize episode rewards
        vt = (vt - vt.mean()) / vt.std()
        self.vt = vt
