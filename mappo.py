import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random

# Set random seeds for reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)  # normalized action probabilities
        )

    def forward(self, x):
        return self.model(x)

class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # scalar value prediction
        )

    def forward(self, x):
        return self.model(x)

class MAPPOAgent:
    def __init__(self, obs_dim, act_dim, num_agents, hidden_dim=128, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.num_agents = num_agents
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.actors = []
        self.optimizers = []
        self.critic = Critic(obs_dim, hidden_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        for _ in range(num_agents):
            actor = Actor(obs_dim, hidden_dim, act_dim)
            self.actors.append(actor)
            self.optimizers.append(optim.Adam(actor.parameters(), lr=lr))

    def select_action(self, obs_batch):
        # Each agent samples an action from its policy
        actions, probs = [], []
        for i, actor in enumerate(self.actors):
            logits = actor(obs_batch[i])
            dist = Categorical(logits)
            action = dist.sample()
            actions.append(action.item())
            probs.append(dist.log_prob(action))  # log-prob needed for PPO update
        return actions, probs

    def compute_returns(self, rewards, dones, last_value):
        # Compute reward-to-go with optional bootstrap
        R = last_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * (1 - dones[step])  # reset if episode ends
            returns.insert(0, R)
        return returns

    def update(self, trajectories):
        total_loss = 0

        for agent_idx in range(self.num_agents):
            # Extract trajectories for this agent
            obs = torch.stack([x['obs'][agent_idx] for x in trajectories])
            actions = torch.tensor([x['actions'][agent_idx] for x in trajectories])
            old_log_probs = torch.stack([x['log_probs'][agent_idx] for x in trajectories])
            rewards = [x['rewards'][agent_idx] for x in trajectories]

            # Baseline value estimates
            values = self.critic(obs).view(-1)
            returns = torch.tensor(self.compute_returns(rewards, [0]*len(rewards), 0.0))
            advantages = returns - values.detach()  # advantage estimation

            # Re-evaluate policy under new actor
            logits = self.actors[agent_idx](obs)
            dist = Categorical(logits)
            new_log_probs = dist.log_prob(actions)

            # PPO clipped objective to stabilize training
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss: value regression
            critic_loss = nn.MSELoss()(values, returns)

            # Update actor network
            self.optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            self.optimizers[agent_idx].step()

            # Update critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            total_loss += (actor_loss + critic_loss).item()

        return total_loss / self.num_agents
