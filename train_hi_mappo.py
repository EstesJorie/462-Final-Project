import torch
import os
import csv
import copy
import random
import numpy as np
from hi_mappo import HiMAPPOAgent
from civilization_env_hi_mappo import CivilizationEnv_HiMAPPO

# ========================================================
# Global Seed for Reproducibility
# Ensures consistent results across runs
# ========================================================
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ========================================================
# User Input for Environment Configuration
# ========================================================
rows, cols = map(int, input("Enter no. of rows and columns: ").split())
num_generations = int(input("Enter number of generations: "))
num_tribes = int(input("Enter number of starting tribes: "))

# ========================================================
# Initialize Environment and Hi-MAPPO Agent
# ========================================================
env = CivilizationEnv_HiMAPPO(rows=rows, cols=cols, num_tribes=num_tribes, seed=42)
obs_dim = rows * cols * 3  # Observation vector size per agent
state_dim = obs_dim        # Global state input to manager
goal_dim = 3               # Number of high-level goals
act_dim = 3                # Number of low-level actions

agent = HiMAPPOAgent(state_dim, obs_dim, goal_dim, act_dim, num_tribes)

# ========================================================
# Training Configuration
# ========================================================
log_interval = 100
score_log = []

# ========================================================
# Training Loop
# ========================================================
for gen in range(1, num_generations + 1):
    obs_raw = env.reset()

    # Prepare global and individual agent observations
    state = torch.tensor(env.get_global_state(), dtype=torch.float32)
    obs_batch = [torch.tensor(o, dtype=torch.float32) for o in env.get_agent_obs()]

    # === High-Level Goal Selection ===
    use_mcts = gen <= 500  # Use MCTS only during early training
    goal_ids, logp_goal = agent.select_goals(state, env=copy.deepcopy(env), use_mcts=use_mcts)

    # === Low-Level Action Selection ===
    actions, logp_actions = agent.select_actions(obs_batch, goal_ids)

    # Store high-level goals in environment for potential reward shaping
    env.last_goals = goal_ids.tolist()

    # === Collect One Trajectory (One Generation) ===
    traj = {
        'state': state,
        'obs': obs_batch,
        'goal': goal_ids,
        'logp_goal': logp_goal,
        'actions': actions,
        'rewards': []
    }

    # Roll out environment for a fixed number of steps
    for step in range(10):
        next_obs_raw, rewards, done, _ = env.step(actions)
        traj['rewards'].append(sum(rewards))  # Sum across all agents

    # === Train Agent using Collected Trajectory ===
    loss = agent.update([traj])
    loss_value = loss if isinstance(loss, float) else loss.item()

    # === Logging & Visualization ===
    if gen % log_interval == 0:
        print(f"\n===== Generation {gen} =====")
        env.render()
        scores = env.compute_final_scores()
        score_log.append((gen, scores))
        print("Loss:", loss_value)

# ========================================================
# Save Score Log to CSV
# ========================================================
with open("hi_mappo_score_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Generation"] + [f"Tribe_{i+1}_Score" for i in range(num_tribes)])
    for generation, scores in score_log:
        writer.writerow([generation] + scores)

# ========================================================
# Save Trained Models
# Each worker and the high-level manager are saved separately
# ========================================================
os.makedirs("trained_models_hi_mappo", exist_ok=True)
for i, net in enumerate(agent.workers):
    torch.save(net.state_dict(), f"trained_models_hi_mappo/worker_{i}.pth")
torch.save(agent.manager.state_dict(), "trained_models_hi_mappo/manager.pth")

print("✅ Hi-MAPPO model and score log saved.")
