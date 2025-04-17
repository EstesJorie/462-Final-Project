import torch
from civilization_env_mappo import CivilizationEnv_MAPPO
from mappo import MAPPOAgent
from GameController import GameController

import random
import numpy as np
import torch

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Get user-defined simulation parameters via controller UI ===
controller = GameController()
rows, cols = controller.getValidDimensions()         # Get valid grid dimensions from user
generations = controller.getValidGenerations()       # Get number of generations (training iterations)
num_tribes = controller.getValidTribeCount()         # Get number of tribes (agents)

# === Initialize environment and MAPPO agent ===
env = CivilizationEnv_MAPPO(rows=rows, cols=cols, num_tribes=num_tribes)
obs_dim = env.rows * env.cols * 3                    # Observation space: each cell has 3 features
act_dim = 3                                          # Action space: assume 3 discrete actions per agent
agent = MAPPOAgent(obs_dim=obs_dim, act_dim=act_dim, num_agents=num_tribes)

# === Training configuration ===
total_iterations = generations                      # Total training episodes/generations
log_interval = 100                                  # Log and render output every 100 iterations

# === Begin MAPPO training loop ===
for iteration in range(1, total_iterations + 1):
    obs_raw = env.reset()                            # Reset the environment
    trajectories = []                                # To store trajectory data for MAPPO update

    # Run a fixed number of steps per episode
    for step in range(10):
        obs_batch = []
        for agent_id in range(env.num_agents):
            flat_obs = torch.tensor(obs_raw.flatten(), dtype=torch.float32)
            obs_batch.append(flat_obs)               # Prepare input for each agent

        actions, log_probs = agent.select_action(obs_batch)  # Select action and record log-probabilities
        next_obs, rewards, done, _ = env.step(actions)       # Take a step in the environment

        # Store transition data for policy update
        trajectories.append({
            'obs': obs_batch,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards
        })

        obs_raw = next_obs

    # After trajectory collection, update MAPPO networks (actor and critic)
    agent.update(trajectories)

    # Logging and optional visualization every log_interval iterations
    if iteration % log_interval == 0:
        print(f"\n========== Generation {iteration} ==========")
        env.render()  # Show the current grid (tribe positions, etc.)

# === Save the trained model parameters ===
import os

save_dir = "trained_models"
os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save each actor network (one per agent) and the shared critic network
for i, actor in enumerate(agent.actors):
    torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_{i}.pth"))
torch.save(agent.critic.state_dict(), os.path.join(save_dir, "critic.pth"))

print("Models saved to 'trained_models/'")
