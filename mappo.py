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

# ============================
# Actor Network
# ============================
# The actor outputs a probability distribution over actions given an observation
class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Fully connected layer from input to hidden layer
            nn.ReLU(),                         # ReLU activation for non-linearity
            nn.Linear(hidden_dim, output_dim), # Fully connected layer to output logits
            nn.Softmax(dim=-1)                 # Normalize outputs into probability distribution
        )

    def forward(self, x):
        return self.model(x)

# ============================
# Critic Network
# ============================
# The critic estimates the value of a given state
class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input to hidden
            nn.ReLU(),                         # Non-linearity
            nn.Linear(hidden_dim, 1)           # Output a single scalar value
        )

    def forward(self, x):
        return self.model(x)

# ============================
# MAPPO Agent
# ============================
# This class handles multiple agents, each with its own actor, and a shared critic.
class MAPPOAgent:
    def __init__(self, obs_dim, act_dim, num_agents, hidden_dim=128, lr=3e-4, gamma=0.99, eps_clip=0.2):
        self.num_agents = num_agents              # Total number of agents
        self.actors = []                          # One actor per agent
        self.optimizers = []                      # Corresponding optimizer per actor
        self.gamma = gamma                        # Discount factor for return computation
        self.eps_clip = eps_clip                  # Clipping threshold for PPO
        self.critic = Critic(obs_dim, hidden_dim) # Shared critic across all agents
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Initialize actor networks and their optimizers
        for _ in range(self.num_agents):
            actor = Actor(obs_dim, hidden_dim, act_dim)
            self.actors.append(actor)
            self.optimizers.append(optim.Adam(actor.parameters(), lr=lr))

    # Select actions for all agents based on their observations
    def select_action(self, obs_batch):
        actions = []  # List of actions chosen
        probs = []    # List of log-probabilities for each action
        for i, actor in enumerate(self.actors):
            logits = actor(obs_batch[i])          # Forward pass for agent i
            dist = Categorical(logits)            # Turn logits into categorical distribution
            action = dist.sample()                # Sample an action
            actions.append(action.item())         # Store chosen action
            probs.append(dist.log_prob(action))   # Store log-probability of chosen action
        return actions, probs

    # Compute discounted returns using the Bellman equation
    def compute_returns(self, rewards, dones, last_value):
        R = last_value              # Start from the bootstrap value
        returns = []
        for step in reversed(range(len(rewards))):
            # If done = 1, future reward does not propagate
            R = rewards[step] + self.gamma * R * (1 - dones[step])
            returns.insert(0, R)    # Insert at the beginning to keep correct order
        return returns

    # Perform PPO update for all agents
    def update(self, trajectories):
        total_loss = 0  # Accumulate loss across all agents for monitoring

        for agent_idx in range(self.num_agents):
            # === Extract per-agent trajectory data ===
            obs = torch.stack([x['obs'][agent_idx] for x in trajectories])              # Observations
            actions = torch.tensor([x['actions'][agent_idx] for x in trajectories])     # Taken actions
            old_log_probs = torch.stack([x['log_probs'][agent_idx] for x in trajectories]) # Log probs
            rewards = [x['rewards'][agent_idx] for x in trajectories]                   # Reward sequence

            # === Compute advantage estimates ===
            values = self.critic(obs).view(-1)    # Value estimates from critic
            returns = torch.tensor(self.compute_returns(rewards, [0]*len(rewards), 0.0)) # Compute returns
            advantages = returns - values.detach() # Detach critic values to prevent backprop

            # === Compute new action probabilities ===
            logits = self.actors[agent_idx](obs)   # Get updated action distribution
            dist = Categorical(logits)
            new_log_probs = dist.log_prob(actions)

            # === PPO clipped objective ===
            ratio = torch.exp(new_log_probs - old_log_probs)  # Importance sampling ratio
            surr1 = ratio * advantages                        # Unclipped objective
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages  # Clipped objective
            actor_loss = -torch.min(surr1, surr2).mean()      # Take conservative estimate (min)

            # === Critic loss (mean squared error) ===
            critic_loss = nn.MSELoss()(values, returns)

            # === Backprop and optimize ===
            self.optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            self.optimizers[agent_idx].step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Accumulate total loss for reporting
            total_loss += (actor_loss + critic_loss).item()

        # Return average loss across all agents
        return total_loss / self.num_agents
