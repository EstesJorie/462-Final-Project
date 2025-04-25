import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Categorical

# Set random seeds for reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ======================================
# High-level policy (Manager)
# Outputs a probability distribution over abstract goals
# ======================================
class HighLevelPolicy(nn.Module):
    def __init__(self, state_dim, goal_dim, hidden_dim=64):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.policy(state)

# ======================================
# Low-level policy (Worker)
# Outputs action probabilities based on agent's observation and goal
# ======================================
class LowLevelPolicy(nn.Module):
    def __init__(self, obs_dim, goal_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, obs, goal_onehot):
        x = torch.cat([obs, goal_onehot], dim=-1)
        return self.policy(x)

# ======================================
# Hi-MAPPO Agent
# Coordinates high-level manager and multiple low-level workers
# ======================================
class HiMAPPOAgent:
    def __init__(self, state_dim, obs_dim, goal_dim, act_dim, num_agents, hidden_dim=64, lr=3e-4):
        self.num_agents = num_agents
        self.goal_dim = goal_dim
        self.manager = HighLevelPolicy(state_dim, goal_dim, hidden_dim)
        self.manager_optim = optim.Adam(self.manager.parameters(), lr=lr)
        self.workers = [LowLevelPolicy(obs_dim, goal_dim, act_dim, hidden_dim) for _ in range(num_agents)]
        self.worker_optims = [optim.Adam(w.parameters(), lr=lr) for w in self.workers]

    def select_goals(self, state):
        probs = self.manager(state)
        dist = Categorical(probs)
        goals = [dist.sample() for _ in range(self.num_agents)]
        logps = [dist.log_prob(g) for g in goals]
        return torch.stack(goals), torch.stack(logps)

    def select_actions(self, obs_batch, goal_ids):
        actions = []
        logps = []
        for i in range(self.num_agents):
            obs = obs_batch[i]
            goal = int(goal_ids[i])
            goal_onehot = torch.nn.functional.one_hot(torch.tensor(goal), num_classes=self.goal_dim).float()
            goal_onehot = goal_onehot.to(obs.device)
            logits = self.workers[i](obs, goal_onehot)
            dist = Categorical(logits=logits)
            action = dist.sample()
            actions.append(action.item())
            logps.append(dist.log_prob(action))
        return actions, logps

    def update(self, trajs):
        states = torch.stack([t['state'] for t in trajs])
        goals = torch.stack([t['goal'] for t in trajs])
        log_probs = torch.stack([t['logp_goal'] for t in trajs])
        rewards = torch.tensor([np.mean(t['rewards']) for t in trajs])
        advantages = rewards - rewards.mean()

        dist = Categorical(self.manager(states))
        new_log_probs = dist.log_prob(goals[:, 0])
        entropy = dist.entropy().mean()  # ✅ 加入 Manager entropy bonus
        ratio = torch.exp(new_log_probs - log_probs[:, 0].detach())

        manager_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 0.8, 1.2) * advantages
        ).mean()

        manager_loss = manager_loss - 0.01 * entropy  # ✅ 加入 entropy 奖励项

        self.manager_optim.zero_grad()
        manager_loss.backward()
        self.manager_optim.step()

        total_worker_loss = 0
        for i in range(self.num_agents):
            obs_i = torch.cat([torch.stack(t['obs'][i]) for t in trajs], dim=0)
            actions_i = torch.tensor([a for t in trajs for a in t['actions'][i]])
            goal_ids_i = torch.tensor([
                t['goal'][i].item() for t in trajs for _ in range(len(t['actions'][i]))
            ])
            goal_onehots_i = torch.nn.functional.one_hot(goal_ids_i, self.goal_dim).float()
            probs = self.workers[i](obs_i, goal_onehots_i)
            dist = Categorical(probs)
            log_probs_i = dist.log_prob(actions_i)
            returns_i = torch.tensor([
                np.mean(t['rewards']) for t in trajs for _ in range(len(t['actions'][i]))
            ])
            worker_loss = -(log_probs_i * (returns_i - returns_i.mean())).mean()
            self.worker_optims[i].zero_grad()
            worker_loss.backward()
            self.worker_optims[i].step()
            total_worker_loss += worker_loss.item()

        return manager_loss.item() + total_worker_loss / self.num_agents