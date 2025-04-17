import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import random

# Set fixed seed for reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Actor network: outputs action probabilities via softmax
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input layer
            nn.ReLU(),                         # Hidden layer activation
            nn.Linear(hidden_dim, output_dim), # Output logits
            nn.Softmax(dim=-1)                 # Convert to probability distribution
        )

    def forward(self, x):
        return self.model(x)

# Critic network: estimates state value (V(s))
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input layer
            nn.ReLU(),                         # Hidden layer activation
            nn.Linear(hidden_dim, 1)           # Single scalar output: value estimate
        )

    def forward(self, x):
        return self.model(x)

# MAPPO agent managing multiple actor networks and a shared critic
class MAPPOAgent:
    def __init__(self, obs_dim, act_dim, num_agents, hidden_dim=128, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.num_agents = num_agents              # Number of agents
        self.actors = []                          # Separate actor network per agent
        self.optimizers = []                      # Optimizers for each actor
        self.gamma = gamma                        # Discount factor
        self.eps_clip = eps_clip                  # PPO clipping threshold
        self.critic = Critic(obs_dim, hidden_dim) # Shared centralized critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize actors and their optimizers
        for _ in range(self.num_agents):
            actor = Actor(obs_dim, hidden_dim, act_dim)
            self.actors.append(actor)
            self.optimizers.append(optim.Adam(actor.parameters(), lr=lr))

    # Select actions for all agents given their observations
    def select_action(self, obs_batch):
        actions = []  # List of chosen actions (ints)
        probs = []    # List of log probabilities for each action
        for i, actor in enumerate(self.actors):
            logits = actor(obs_batch[i])          # Forward pass through actor network
            dist = Categorical(logits)            # Treat output as categorical distribution
            action = dist.sample()                # Sample action from distribution
            actions.append(action.item())         # Store action
            probs.append(dist.log_prob(action))   # Store log probability of action
        return actions, probs

    # Compute discounted returns for a trajectory
    def compute_returns(self, rewards, dones, last_value):
        R = last_value              # Bootstrap from final state value
        returns = []
        for step in reversed(range(len(rewards))):
            # Bellman equation for return
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)    # Prepend to maintain order
        return returns

    # Update actor and critic networks using PPO loss
    def update(self, trajectories):
        for agent_idx in range(self.num_agents):
            # Extract trajectory data for this agent
            obs = torch.stack([x['obs'][agent_idx] for x in trajectories])                  # Shape: [T, obs_dim]
            actions = torch.tensor([x['actions'][agent_idx] for x in trajectories])         # Shape: [T]
            old_log_probs = torch.stack([x['log_probs'][agent_idx] for x in trajectories])  # Shape: [T]
            rewards = [x['rewards'][agent_idx] for x in trajectories]                       # Shape: [T]

            # Compute state values and returns
            values = self.critic(obs).view(-1)                                              # Estimated V(s)
            returns = torch.tensor(self.compute_returns(rewards, [0]*len(rewards), 0.0))    # Ground-truth returns
            advantages = returns - values.detach()                                          # Advantage estimates

            # === PPO Update ===
            logits = self.actors[agent_idx](obs)            # Get new action probs
            dist = Categorical(logits)
            new_log_probs = dist.log_prob(actions)

            # Importance sampling ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()    # PPO policy loss (maximize clipped advantage)

            # Critic loss: MSE between estimated and actual returns
            critic_loss = nn.MSELoss()(values, returns)

            # === Backpropagation ===
            self.optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            self.optimizers[agent_idx].step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
