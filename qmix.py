import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# Set random seed for reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =====================================
# Individual Q-network for each agent
# =====================================
class AgentNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # Input layer for agent's observation
            nn.ReLU(),                         # Non-linear activation
            nn.Linear(hidden_dim, output_dim)  # Output Q-values for each possible action
        )

    def forward(self, x):
        return self.net(x)

# =====================================
# Mixer Network: combines individual Qs into a joint Q
# Uses hypernetworks conditioned on global state
# =====================================
class MixerNet(nn.Module):
    def __init__(self, n_agents, state_dim, hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        # Hypernetworks to produce mixing weights/biases from global state
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)  # Weights for first layer
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)             # Weights for second layer
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)             # Bias for first layer
        self.hyper_b2 = nn.Sequential(                               # Bias for final output
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_qs, state):
        bs = agent_qs.size(0)          # Batch size
        state = state.view(bs, -1)     # Flatten global state

        # Hypernetwork-generated weights and biases
        w1 = self.hyper_w1(state).view(bs, self.n_agents, -1)  # Shape: [B, n_agents, hidden]
        b1 = self.hyper_b1(state).view(bs, 1, -1)               # Shape: [B, 1, hidden]

        # First mixing layer: agent Qs * weights + bias
        hidden = torch.bmm(agent_qs.view(bs, 1, -1), w1) + b1   # Shape: [B, 1, hidden]
        hidden = torch.relu(hidden)

        # Second mixing layer: hidden * weights + bias
        w2 = self.hyper_w2(state).view(bs, -1, 1)               # Shape: [B, hidden, 1]
        b2 = self.hyper_b2(state).view(bs, 1, 1)                # Shape: [B, 1, 1]
        y = torch.bmm(hidden, w2) + b2                          # Final output: [B, 1, 1]
        return y.view(-1, 1)                                    # Reshape to [B, 1]

# =====================================
# QMIX Agent
# Learns agent Q-networks and combines them via the mixer
# =====================================
class QMIXAgent:
    def __init__(
        self, obs_dim, state_dim, act_dim, n_agents,
        hidden_dim=64, buffer_size=10000, batch_size=64,
        lr=1e-3, gamma=0.99
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size

        # Initialize online and target Q-networks for each agent
        self.agent_nets = [AgentNet(obs_dim, hidden_dim, act_dim) for _ in range(n_agents)]
        self.target_agent_nets = [AgentNet(obs_dim, hidden_dim, act_dim) for _ in range(n_agents)]

        # Initialize online and target mixer networks
        self.mix_net = MixerNet(n_agents, state_dim, hidden_dim=160)
        self.target_mix_net = MixerNet(n_agents, state_dim, hidden_dim=160)

        # Sync target networks
        for i in range(n_agents):
            self.target_agent_nets[i].load_state_dict(self.agent_nets[i].state_dict())
        self.target_mix_net.load_state_dict(self.mix_net.state_dict())

        # Optimizers
        self.agent_opts = [optim.Adam(net.parameters(), lr=lr) for net in self.agent_nets]
        self.mixer_opt = optim.Adam(self.mix_net.parameters(), lr=lr)

        # Replay buffer to store transitions
        self.buffer = deque(maxlen=buffer_size)

    # ε-greedy policy for action selection
    def select_actions(self, obs_batch, epsilon=0.05):
        actions = []
        for i in range(self.n_agents):
            q_values = self.agent_nets[i](obs_batch[i])  # Get Q-values for current obs
            if random.random() < epsilon:
                action = random.randint(0, self.act_dim - 1)  # Explore
            else:
                action = q_values.argmax().item()  # Exploit best action
            actions.append(action)
        return actions

    # Store a transition (one timestep) in buffer
    def store_transition(self, obs, state, actions, rewards, next_obs, next_state):
        self.buffer.append((obs, state, actions, rewards, next_obs, next_state))

    # Train all networks using a batch of experiences from buffer
    def train(self):
        if len(self.buffer) < self.batch_size:
            return  # Wait until buffer has enough samples

        # Sample a batch
        batch = random.sample(self.buffer, self.batch_size)
        obs_batch, state_batch, actions_batch, rewards_batch, next_obs_batch, next_state_batch = zip(*batch)

        # Convert to tensors
        obs_batch = torch.stack([torch.stack(x) for x in obs_batch])             # [B, n_agents, obs_dim]
        next_obs_batch = torch.stack([torch.stack(x) for x in next_obs_batch])   # [B, n_agents, obs_dim]
        actions_batch = torch.tensor(actions_batch)                              # [B, n_agents]
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)         # [B, n_agents]
        state_batch = torch.stack(state_batch)                                   # [B, state_dim]
        next_state_batch = torch.stack(next_state_batch)                         # [B, state_dim]

        # Compute individual agent Q-values
        agent_qs = []
        target_qs = []
        for i in range(self.n_agents):
            # Q-values for taken actions
            q_eval = self.agent_nets[i](obs_batch[:, i, :])  # [B, act_dim]
            q_eval = q_eval.gather(1, actions_batch[:, i].unsqueeze(1)).squeeze(1)  # [B]
            agent_qs.append(q_eval)

            # Max Q-values for next state (target network)
            q_target = self.target_agent_nets[i](next_obs_batch[:, i, :])  # [B, act_dim]
            max_q_target = q_target.max(dim=1)[0]                          # [B]
            target_qs.append(max_q_target)

        agent_qs = torch.stack(agent_qs, dim=1)       # [B, n_agents]
        target_qs = torch.stack(target_qs, dim=1)     # [B, n_agents]

        # Combine agent Qs into global Q via mixer
        q_total_eval = self.mix_net(agent_qs, state_batch)                            # [B, 1]
        q_total_target = self.target_mix_net(target_qs, next_state_batch).detach()    # [B, 1]

        # Compute target using sum of individual rewards + future value
        targets = rewards_batch.sum(dim=1, keepdim=True) + self.gamma * q_total_target

        # MSE loss between predicted joint Q and target
        loss = nn.MSELoss()(q_total_eval, targets)

        # Backprop and optimize
        self.mixer_opt.zero_grad()
        for opt in self.agent_opts:
            opt.zero_grad()
        loss.backward()
        self.mixer_opt.step()
        for opt in self.agent_opts:
            opt.step()

        return loss

    # Update target networks to match current networks
    def update_targets(self):
        for i in range(self.n_agents):
            self.target_agent_nets[i].load_state_dict(self.agent_nets[i].state_dict())
        self.target_mix_net.load_state_dict(self.mix_net.state_dict())
