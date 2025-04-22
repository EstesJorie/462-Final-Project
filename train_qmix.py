import torch
import os
import numpy as np
from civilization_env_qmix import CivilizationEnv_QMIX
from qmix import QMIXAgent
from GameController import GameController
import random
import csv

# =======================================
# Set Random Seed for Reproducibility
# =======================================
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =======================================
# Game Configuration via GameController UI
# =======================================
controller = GameController()
rows, cols = controller.getValidDimensions()           # Grid dimensions
generations = controller.getValidGenerations()         # Total training iterations
num_tribes = controller.getValidTribeCount()           # Number of agents

# =======================================
# Environment and QMIX Agent Initialization
# =======================================
env = CivilizationEnv_QMIX(rows=rows, cols=cols, num_tribes=num_tribes)
obs_dim = rows * cols * 3     # Flattened grid observation
state_dim = obs_dim           # Shared global state input to mixer
act_dim = 3                   # Discrete action space per agent
agent = QMIXAgent(obs_dim, state_dim, act_dim, n_agents=num_tribes)

total_iterations = generations
log_interval = 100            # Interval for logging/rendering

# =======================================
# Utility: Convert raw obs to tensor batch
# =======================================
def preprocess_obs(obs_raw):
    return [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]

# =======================================
# Logging Structures
# =======================================
score_log = []
loss_log = []

# =======================================
# Training Loop
# =======================================
for iteration in range(1, total_iterations + 1):
    obs_raw = env.reset()
    state_raw = torch.tensor(obs_raw.flatten(), dtype=torch.float32)

    # Run an episode (fixed steps)
    for step in range(20):
        obs_batch = preprocess_obs(obs_raw)

        # ε-greedy exploration schedule
        epsilon = max(0.05, 1 - iteration / 2000)

        # Agents choose actions
        actions = agent.select_actions(obs_batch, epsilon=epsilon)

        # Environment executes actions
        next_obs_raw, rewards, done, _ = env.step(actions)
        next_state_raw = torch.tensor(next_obs_raw.flatten(), dtype=torch.float32)
        next_obs_batch = preprocess_obs(next_obs_raw)

        # Store transition in replay buffer
        agent.store_transition(obs_batch, state_raw, actions, rewards, next_obs_batch, next_state_raw)

        # Update obs/state for next step
        obs_raw = next_obs_raw
        state_raw = next_state_raw

    # Train from buffer
    loss = agent.train()
    loss_value = loss.item() if loss is not None else None
    if loss_value is not None:
        loss_log.append((iteration, loss_value))

    # Periodically update target networks
    if iteration % 200 == 0:
        agent.update_targets()

    # Log environment state and reward
    if iteration % log_interval == 0:
        print(f"\n========== Generation {iteration} ==========")
        env.render()
        scores = env.compute_final_scores()
        score_log.append((iteration, scores))
        if loss_value is not None:
            print("Loss:", loss_value)

# =======================================
# Save Trained Models
# One file per agent and one for the mixer
# =======================================
save_dir = "trained_models_qmix"
os.makedirs(save_dir, exist_ok=True)
for i, net in enumerate(agent.agent_nets):
    torch.save(net.state_dict(), os.path.join(save_dir, f"qmix_agent_{i}.pth"))
torch.save(agent.mix_net.state_dict(), os.path.join(save_dir, "qmix_mixer.pth"))
print("QMIX model has been saved to 'trained_models_qmix/'")

# =======================================
# Save Training Score Log as CSV
# =======================================
with open("qmix_score_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Generation"] + [f"Tribe_{i + 1}_Score" for i in range(num_tribes)])
    for generation, scores in score_log:
        writer.writerow([generation] + scores)

print("✅ Score logs saved.")
