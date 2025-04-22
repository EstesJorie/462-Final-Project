import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Categorical
from mcts_hi_mappo import MCTS  # Import the MCTS class for high-level goal planning

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
        # Simple feedforward network: input is global state, output is goal probabilities
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
            nn.Softmax(dim=-1)  # Output softmax probabilities over possible goals
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
        # The input is a concatenation of observation and goal one-hot encoding
        self.policy = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)  # Output action probabilities
        )

    def forward(self, obs, goal_onehot):
        # Concatenate observation and goal before feeding into the policy
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

        # Manager policy (shared across all agents)
        self.manager = HighLevelPolicy(state_dim, goal_dim, hidden_dim)
        self.manager_optim = optim.Adam(self.manager.parameters(), lr=lr)

        # Create one worker policy and optimizer per agent
        self.workers = [LowLevelPolicy(obs_dim, goal_dim, act_dim, hidden_dim) for _ in range(num_agents)]
        self.worker_optims = [optim.Adam(w.parameters(), lr=lr) for w in self.workers]

    # Select high-level goals for all agents
    def select_goals(self, state, env=None, use_mcts=False):
        if use_mcts and env is not None:
            # Use MCTS to select a single shared goal (centralized planning)
            mcts = MCTS(self, env, state)
            goal = mcts.run()
            return torch.tensor([goal] * self.num_agents), torch.tensor([0.0] * self.num_agents)
        else:
            # Use the manager network to sample separate goals for each agent
            probs = self.manager(state)
            dist = Categorical(probs)
            goals = [dist.sample() for _ in range(self.num_agents)]
            logps = [dist.log_prob(g) for g in goals]
            return torch.stack(goals), torch.stack(logps)

    # Select actions for each agent using their worker policy
    def select_actions(self, obs_batch, goal_ids):
        actions = []
        logps = []
        for i in range(self.num_agents):
            obs = obs_batch[i]
            goal = int(goal_ids[i])
            # Convert scalar goal index to one-hot tensor
            goal_onehot = torch.nn.functional.one_hot(torch.tensor(goal), num_classes=self.goal_dim).float()
            goal_onehot = goal_onehot.to(obs.device)

            # Get action distribution from worker policy
            logits = self.workers[i](obs, goal_onehot)
            dist = Categorical(logits=logits)
            action = dist.sample()
            actions.append(action.item())
            logps.append(dist.log_prob(action))
        return actions, logps

    # Update the manager and workers using collected trajectory data
    def update(self, trajs):
        # === Update Manager ===
        states = torch.stack([t['state'] for t in trajs])         # States at goal selection
        goals = torch.stack([t['goal'] for t in trajs])           # Selected goals
        log_probs = torch.stack([t['logp_goal'] for t in trajs])  # Log prob of selected goals
        rewards = torch.tensor([sum(t['rewards']) for t in trajs]) # Total reward from trajectory

        # Compute normalized advantage
        advantages = rewards - rewards.mean()

        # New log_probs using current policy
        dist = Categorical(self.manager(states))
        new_log_probs = dist.log_prob(goals[:, 0])  # All agents used the same goal index

        # Compute PPO ratio
        ratio = torch.exp(new_log_probs - log_probs[:, 0].detach())

        # PPO objective with clipping
        manager_loss = -torch.min(
            ratio * advantages,
            torch.clamp(ratio, 0.8, 1.2) * advantages
        ).mean()

        # Update manager
        self.manager_optim.zero_grad()
        manager_loss.backward()
        self.manager_optim.step()

        # === Update Workers ===
        total_worker_loss = 0
        for i in range(self.num_agents):
            obs = torch.stack([t['obs'][i] for t in trajs])  # Observations for agent i
            act = torch.tensor([t['actions'][i] for t in trajs])  # Taken actions
            goal_ids = goals[:, i]  # Corresponding goals for agent i
            goal_onehots = torch.nn.functional.one_hot(goal_ids, self.goal_dim).float()

            # Compute action probabilities
            probs = self.workers[i](obs, goal_onehots)
            dist = Categorical(probs)
            log_probs = dist.log_prob(act)
            returns = rewards  # Shared reward for simplicity

            # Compute REINFORCE-style policy loss with baseline
            worker_loss = -(log_probs * (returns - returns.mean())).mean()

            # Update worker policy
            self.worker_optims[i].zero_grad()
            worker_loss.backward()
            self.worker_optims[i].step()

            total_worker_loss += worker_loss.item()

        # Return combined loss for logging
        total_loss = manager_loss.item() + total_worker_loss / self.num_agents
        return total_loss
