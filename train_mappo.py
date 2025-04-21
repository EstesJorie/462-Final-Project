import torch
from civilization_env_mappo import CivilizationEnv_MAPPO
from mappo import MAPPOAgent
from GameController import GameController
import random
import numpy as np
import os

# ======================================
# Set deterministic behavior for reproducibility
# ======================================
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================
# Initialize game controller to get user-defined parameters
# ======================================
controller = GameController()
rows, cols = controller.getValidDimensions()     # Get grid dimensions
generations = controller.getValidGenerations()   # Total training steps
num_tribes = controller.getValidTribeCount()     # Number of agents (tribes)

# ======================================
# Initialize environment and MAPPO agent
# ======================================
env = CivilizationEnv_MAPPO(rows=rows, cols=cols, num_tribes=num_tribes)

obs_dim = env.rows * env.cols * 3   # Flattened observation size: pop, food, tribe_id
act_dim = 3                         # Each agent has 3 possible actions
agent = MAPPOAgent(obs_dim=obs_dim, act_dim=act_dim, num_agents=num_tribes)

# ======================================
# Training parameters
# ======================================
total_iterations = generations     # Number of training episodes
log_interval = 100                 # Log and render every N iterations

# Score tracking
score_log = []

# ======================================
# Training Loop
# Each iteration simulates one generation of environment interaction
# ======================================
for iteration in range(1, total_iterations + 1):
    obs_raw = env.reset()          # Reset environment
    trajectories = []              # Collect transition data for one episode

    # Simulate one episode (fixed length)
    for step in range(10):
        obs_batch = []
        for agent_id in range(env.num_agents):
            flat_obs = torch.tensor(obs_raw.flatten(), dtype=torch.float32)  # Use full map as input
            obs_batch.append(flat_obs)

        # Agents select actions and get log probabilities
        actions, log_probs = agent.select_action(obs_batch)
        next_obs, rewards, done, _ = env.step(actions)

        # Record transition for PPO update
        trajectories.append({
            'obs': obs_batch,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards
        })

        obs_raw = next_obs  # Move to next state

    # Update MAPPO agent using collected data
    loss = agent.update(trajectories)

    # ======================================
    # Logging and visualization
    # ======================================
    if iteration % log_interval == 0:
        print(f"\n========== Generation {iteration} ==========")
        env.render()                       # Show grid and debug info
        scores = env.compute_final_scores()
        score_log.append((iteration, scores))
        print("Loss:", loss)

# ======================================
# Save trained actor and critic models
# Each agent's actor is saved separately
# ======================================
os.makedirs("trained_models_mappo", exist_ok=True)
for i, actor in enumerate(agent.actors):
    torch.save(actor.state_dict(), f"trained_models_mappo/actor_{i}.pth")
torch.save(agent.critic.state_dict(), "trained_models_mappo/critic.pth")
print("✅ Models saved to 'trained_models_mappo/'")

# ======================================
# Save training performance scores to CSV
# Columns: Generation, Tribe_1_Score, Tribe_2_Score, ...
# ======================================
import csv
with open("mappo_score_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Generation"] + [f"Tribe_{i+1}_Score" for i in range(num_tribes)])
    for generation, scores in score_log:
        writer.writerow([generation] + scores)

print("✅ Score log saved to 'mappo_score_log.csv'")
