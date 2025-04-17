import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Categorical

# Set random seed for reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# === High-Level Policy Network (Goal Selector) ===
class HighLevelPolicy(nn.Module):
    def __init__(self, state_dim, goal_dim, hidden_dim=64):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # Input: global state
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),   # Output: logits over goal space
            nn.Softmax(dim=-1)                 # Convert to goal probability distribution
        )

    def forward(self, state):
        return self.policy(state)

# === Low-Level Worker Policy (Per Agent) ===
class LowLevelPolicy(nn.Module):
    def __init__(self, obs_dim, goal_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.policy = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),  # Input: local obs + one-hot goal
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),             # Output: action logits
            nn.Softmax(dim=-1)                          # Convert to action probabilities
        )

    def forward(self, obs, goal_onehot):
        x = torch.cat([obs, goal_onehot], dim=-1)       # Concatenate observation and goal
        return self.policy(x)

# === Hierarchical MAPPO Controller ===
class HiMAPPOAgent:
    def __init__(
        self, state_dim, obs_dim, goal_dim, act_dim,
        num_agents, hidden_dim=64, lr=3e-4
    ):
        self.num_agents = num_agents
        self.goal_dim = goal_dim

        # High-level manager (shared across all agents)
        self.manager = HighLevelPolicy(state_dim, goal_dim, hidden_dim)
        self.manager_optim = optim.Adam(self.manager.parameters(), lr=lr)

        # Each agent has its own low-level worker policy
        self.workers = [LowLevelPolicy(obs_dim, goal_dim, act_dim, hidden_dim) for _ in range(num_agents)]
        self.worker_optims = [optim.Adam(w.parameters(), lr=lr) for w in self.workers]

    # High-level goal selection for each agent (same manager used for all)
    def select_goals(self, state):
        probs = self.manager(state)                        # [goal_dim] probability distribution
        dist = Categorical(probs)
        goals = [dist.sample() for _ in range(self.num_agents)]       # One goal per agent
        logps = [dist.log_prob(g) for g in goals]                     # Log-probabilities for training
        return torch.stack(goals), torch.stack(logps)

    # Low-level action selection for each agent, given goal
    def select_actions(self, obs_batch, goals):
        actions, log_probs = [], []
        for i, obs in enumerate(obs_batch):
            goal = goals[i]
            goal_onehot = torch.nn.functional.one_hot(goal, self.goal_dim).float()
            probs = self.workers[i](obs, goal_onehot)
            dist = Categorical(probs)
            action = dist.sample()
            actions.append(action.item())
            log_probs.append(dist.log_prob(action))
        return actions, log_probs

    # Update both manager and workers using trajectories
    def update(self, trajs):
        # === Manager Update ===
        states = torch.stack([t['state'] for t in trajs])                # [B, state_dim]
        goals = torch.stack([t['goal'] for t in trajs])                  # [B, num_agents]
        log_probs = torch.stack([t['logp_goal'] for t in trajs])         # [B, num_agents]
        rewards = torch.tensor([sum(t['rewards']) for t in trajs])      # [B] total reward per trajectory

        advantages = rewards - rewards.mean()
        dist = Categorical(self.manager(states))                         # Recalculate dist over goals
        new_log_probs = dist.log_prob(goals[:, 0])                       # Only use first agent's goal log-prob (simplification)
        ratio = torch.exp(new_log_probs - log_probs[:, 0].detach())     # PPO importance ratio
        loss = -torch.min(                                              # PPO clipped loss for manager
            ratio * advantages,
            torch.clamp(ratio, 0.8, 1.2) * advantages
        ).mean()

        self.manager_optim.zero_grad()
        loss.backward()
        self.manager_optim.step()

        # === Worker Policy Update (each agent individually) ===
        for i in range(self.num_agents):
            obs = torch.stack([t['obs'][i] for t in trajs])              # [B, obs_dim]
            act = torch.tensor([t['actions'][i] for t in trajs])         # [B]
            goal_ids = goals[:, i]                                       # [B]
            goal_onehots = torch.nn.functional.one_hot(goal_ids, self.goal_dim).float()  # [B, goal_dim]

            probs = self.workers[i](obs, goal_onehots)
            dist = Categorical(probs)
            log_probs = dist.log_prob(act)

            returns = torch.tensor([sum(t['rewards']) for t in trajs])   # Simplified return = sum of rewards
            loss = -(log_probs * (returns - returns.mean())).mean()     # REINFORCE with baseline (mean return)

            self.worker_optims[i].zero_grad()
            loss.backward()
            self.worker_optims[i].step()
