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


class AgentNet(nn.Module):
    # Individual Q-network for each agent
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MixerNet(nn.Module):
    # Mixer network: combines individual Qs into a global Q using hypernetworks
    def __init__(self, n_agents, state_dim, mixer_hidden_dim=32):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim

        self.hyper_w1 = nn.Linear(state_dim, n_agents * mixer_hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixer_hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, mixer_hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixer_hidden_dim),
            nn.ReLU(),
            nn.Linear(mixer_hidden_dim, 1)
        )

    def forward(self, agent_qs, state):
        bs = agent_qs.size(0)
        state = state.view(bs, -1)

        w1 = self.hyper_w1(state).view(bs, self.n_agents, -1)
        b1 = self.hyper_b1(state).view(bs, 1, -1)

        hidden = torch.bmm(agent_qs.view(bs, 1, -1), w1) + b1
        hidden = torch.relu(hidden)

        w2 = self.hyper_w2(state).view(bs, -1, 1)
        b2 = self.hyper_b2(state).view(bs, 1, 1)

        y = torch.bmm(hidden, w2) + b2
        return y.view(-1, 1)


class QMIXAgent:
    # QMIX agent: trains agent Q-nets and a mixer to estimate joint Q-values
    def __init__(
            self, obs_dim, state_dim, act_dim, n_agents,
            hidden_dim=64, mixer_hidden_dim=200,
            buffer_size=10000, batch_size=64,
            lr=1e-3, gamma=0.99
    ):
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.batch_size = batch_size

        self.agent_nets = [AgentNet(obs_dim, hidden_dim, act_dim) for _ in range(n_agents)]
        self.target_agent_nets = [AgentNet(obs_dim, hidden_dim, act_dim) for _ in range(n_agents)]

        self.mix_net = MixerNet(n_agents, state_dim, mixer_hidden_dim=mixer_hidden_dim)
        self.target_mix_net = MixerNet(n_agents, state_dim, mixer_hidden_dim=mixer_hidden_dim)

        for i in range(n_agents):
            self.target_agent_nets[i].load_state_dict(self.agent_nets[i].state_dict())
        self.target_mix_net.load_state_dict(self.mix_net.state_dict())

        self.agent_opts = [optim.Adam(net.parameters(), lr=lr) for net in self.agent_nets]
        self.mixer_opt = optim.Adam(self.mix_net.parameters(), lr=lr)

        self.buffer = deque(maxlen=buffer_size)

    def select_actions(self, obs_batch, epsilon=0.05):
        """
        Select actions for each agent using epsilon-greedy exploration.

        Args:
            obs_batch: list of observations for each agent
            epsilon: probability of choosing a random action
        Returns:
            list of selected actions
        """
        if isinstance(obs_batch[0], torch.Tensor):
            obs_tensor = torch.stack(obs_batch)
        else:
            obs_tensor = torch.FloatTensor(obs_batch)

        actions = []
        with torch.no_grad():
            for i in range(self.n_agents):
                q_values = self.agent_nets[i](obs_tensor[i])
                if random.random() < epsilon:
                    action = random.randint(0, self.act_dim - 1)
                else:
                    action = q_values.argmax().item()
                actions.append(action)
        return actions

    def store_transition(self, obs, state, actions, rewards, next_obs, next_state):
        # store a transition tuple into the replay buffer
        self.buffer.append((obs, state, actions, rewards, next_obs, next_state))

    def train(self):
        """
        Sample a batch from the replay buffer and update networks.

        - compute target joint Q-values
        - minimize TD error between predicted and target joint Q
        """
        if len(self.buffer) < self.batch_size:
            return  # not enough data to train

        batch = random.sample(self.buffer, self.batch_size)
        obs_batch, state_batch, actions_batch, rewards_batch, next_obs_batch, next_state_batch = zip(*batch)

        obs_batch = torch.stack([torch.stack(x) for x in obs_batch])
        next_obs_batch = torch.stack([torch.stack(x) for x in next_obs_batch])
        actions_batch = torch.tensor(actions_batch)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32)
        state_batch = torch.stack(state_batch)
        next_state_batch = torch.stack(next_state_batch)

        agent_qs, target_qs = [], []
        for i in range(self.n_agents):
            q_eval = self.agent_nets[i](obs_batch[:, i, :])
            q_eval = q_eval.gather(1, actions_batch[:, i].unsqueeze(1)).squeeze(1)
            agent_qs.append(q_eval)

            q_target = self.target_agent_nets[i](next_obs_batch[:, i, :])
            max_q_target = q_target.max(dim=1)[0]
            target_qs.append(max_q_target)

        agent_qs = torch.stack(agent_qs, dim=1)
        target_qs = torch.stack(target_qs, dim=1)

        q_total_eval = self.mix_net(agent_qs, state_batch)
        q_total_target = self.target_mix_net(target_qs, next_state_batch).detach()

        targets = rewards_batch.sum(dim=1, keepdim=True) + self.gamma * q_total_target

        loss = nn.MSELoss()(q_total_eval, targets)

        self.mixer_opt.zero_grad()
        for opt in self.agent_opts:
            opt.zero_grad()
        loss.backward()
        self.mixer_opt.step()
        for opt in self.agent_opts:
            opt.step()

        return loss

    def update_targets(self):
        """
        Soft update target networks to match current networks.
        """
        for i in range(self.n_agents):
            self.target_agent_nets[i].load_state_dict(self.agent_nets[i].state_dict())
        self.target_mix_net.load_state_dict(self.mix_net.state_dict())
