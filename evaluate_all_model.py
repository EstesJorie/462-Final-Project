import torch
import random
import matplotlib.pyplot as plt
import os
import gc 
import pandas as pd
import numpy as np

# === Custom Environments and Agents ===
from civilization_env_mappo import CivilizationEnv_MAPPO
from civilization_env_qmix import CivilizationEnv_QMIX
from civilization_env_hi_mappo import CivilizationEnv_HiMAPPO
from civilization_env_hi_mappo_no_mcts import CivilizationEnv_HiMAPPO as CivilizationEnv_HiMAPPO_No_MCTS
from mappo import MAPPOAgent
from qmix import QMIXAgent
from hi_mappo import HiMAPPOAgent
from CivilisationSimulation2 import CivilisationSimulation  # Random baseline

# ======================================
# Seed for Reproducibility
# ======================================
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ======================================
# Utility: Count how many cells are occupied by each tribe
# Used for tracking territory size
# ======================================
def count_tribe_cells(grid):
    tribe_cells = {}
    for row in grid:
        for cell in row:
            if cell.tribe:
                tribe_cells[cell.tribe] = tribe_cells.get(cell.tribe, 0) + 1
    return [tribe_cells.get(i + 1, 0) for i in range(3)]

# ======================================
# Utility: Moving average smoothing
# Used to smooth curves for better visualization
# ======================================
def smooth(data, window=50):
    return pd.Series(data).rolling(window, min_periods=1).mean()

# ======================================
# Main Evaluation Function
# This function evaluates:
#   - Trained MAPPO agent
#   - Trained QMIX agent
#   - Trained Hi-MAPPO agent
#   - Random baseline
# over multiple episodes, logs their performance, and plots results.
# ======================================
def evaluate_all(rows, cols, num_tribes, num_episodes=1000, log_interval=100):
    obs_dim = rows * cols * 3
    act_dim = 3
    goal_dim = 3
    steps_per_episode = 100  # Fixed number of environment steps per evaluation episode

    # === Load trained MAPPO model ===
    mappo_env = CivilizationEnv_MAPPO(rows, cols, num_tribes)
    mappo_agent = MAPPOAgent(obs_dim, act_dim, num_tribes)
    for i, actor in enumerate(mappo_agent.actors):
        actor.load_state_dict(torch.load(os.path.join("trained_models_mappo", f"actor_{i}.pth")))
    mappo_agent.critic.load_state_dict(torch.load(os.path.join("trained_models_mappo", "critic.pth")))

    # === Load trained QMIX model ===
    qmix_env = CivilizationEnv_QMIX(rows, cols, num_tribes)
    state_dim = (rows * cols * 3) 
    agent_dim = rows * cols * 3  
    action_dim = 3
    
    qmix_agent = QMIXAgent(
        obs_dim=agent_dim,
        state_dim=state_dim,
        act_dim=action_dim,
        n_agents=num_tribes,   
        hidden_dim=64,        # Agent network hidden dim
        mixer_hidden_dim=250, # Mixer network hidden dim
        buffer_size=10000,
        batch_size=64,
        lr=1e-3,
        gamma=0.99
    )
    print(f"Agent Network Input dim: {agent_dim}")
    print(f"State dim: {state_dim}")
    print("Mixer Network dimensions:")
    print(f"- Input dim: {state_dim}")
    print(f"- Hidden dim: {qmix_agent.mixer_hidden_dim}")

    try:
        for i, net in enumerate(qmix_agent.agent_nets):
            net.load_state_dict(torch.load(os.path.join("trained_models_qmix", f"qmix_agent_{i}.pth")))
        qmix_agent.mix_net.load_state_dict(torch.load(os.path.join("trained_models_qmix", "qmix_mixer.pth")))
    except RuntimeError as e:
        print(f"Error loading QMIX model: {e}")
        print("Please ensure the saved model architecture matches the current one.")
        raise

    # === Load trained Hi-MAPPO model ===
    hi_env = CivilizationEnv_HiMAPPO(rows, cols, num_tribes)
    hi_agent = HiMAPPOAgent(obs_dim, obs_dim, goal_dim, act_dim, num_tribes)
    for i, worker in enumerate(hi_agent.workers):
        worker.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo", f"worker_{i}.pth")))
    hi_agent.manager.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo", "manager.pth")))

    # === Initialize Hi-MAPPO No MCTS model ===
    hi_env_no_mcts = CivilizationEnv_HiMAPPO_No_MCTS(rows, cols, num_tribes)
    hi_agent_no_mcts = HiMAPPOAgent(obs_dim, obs_dim, goal_dim, act_dim, num_tribes)
    for i, worker in enumerate(hi_agent_no_mcts.workers):
        worker.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo_no_mcts", f"worker_{i}.pth")))
    hi_agent_no_mcts.manager.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo_no_mcts", "manager.pth")))

    # === Initialize logs to record population, food, occupied cells, final scores ===
    mappo_pops, qmix_pops, hi_pops, rand_pops = [], [], [], []
    mappo_foods, qmix_foods, hi_foods, rand_foods = [], [], [], []
    mappo_cells, qmix_cells, hi_cells, rand_cells = [], [], [], []
    mappo_scores, qmix_scores, hi_scores, rand_scores = [], [], [], []
    hi_no_mcts_pops, hi_no_mcts_foods, hi_no_mcts_cells, hi_no_mcts_scores = [], [], [], []

    # ======================================
    # Evaluation Loop
    # ======================================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    for episode in range(1, num_episodes + 1):

        # --- Evaluate MAPPO ---
        obs_raw = mappo_env.reset()
        for _ in range(steps_per_episode):
            obs_batch = [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]
            actions, _ = mappo_agent.select_action(obs_batch)
            obs_raw, _, _, _ = mappo_env.step(actions)
        mappo_scores.append(sum(mappo_env.compute_final_scores()))
        mappo_pops.append(sum(cell.population for row in mappo_env.sim.grid for cell in row if cell.tribe))
        mappo_foods.append(sum(cell.food for row in mappo_env.sim.grid for cell in row if cell.tribe))
        mappo_cells.append(sum(count_tribe_cells(mappo_env.sim.grid)))

        # --- Evaluate QMIX ---
        obs_raw = qmix_env.reset()
        state = torch.tensor(obs_raw.flatten(), dtype=torch.float32)
        for _ in range(steps_per_episode):
            obs_batch = [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]
            actions = qmix_agent.select_actions(obs_batch, epsilon=0.0)  # Greedy
            obs_raw, _, _, _ = qmix_env.step(actions)
            state = torch.tensor(obs_raw.flatten(), dtype=torch.float32)
        qmix_scores.append(sum(qmix_env.compute_final_scores()))
        qmix_pops.append(sum(cell.population for row in qmix_env.sim.grid for cell in row if cell.tribe))
        qmix_foods.append(sum(cell.food for row in qmix_env.sim.grid for cell in row if cell.tribe))
        qmix_cells.append(sum(count_tribe_cells(qmix_env.sim.grid)))

        # --- Evaluate Hi-MAPPO ---
        obs_raw = hi_env.reset()
        state = torch.tensor(hi_env.get_global_state(), dtype=torch.float32)
        for _ in range(steps_per_episode):
            obs_batch = [torch.tensor(o, dtype=torch.float32) for o in hi_env.get_agent_obs()]
            goals, _ = hi_agent.select_goals(state)

        # === Hi-MAPPO No MCTS Evaluation ===
        obs_raw = hi_env_no_mcts.reset()
        state = torch.tensor(hi_env_no_mcts.get_global_state(), dtype=torch.float32)
        obs_batch = [torch.tensor(o, dtype=torch.float32) for o in hi_env_no_mcts.get_agent_obs()]
        for _ in range(steps_per_episode):
            goals, _ = hi_agent_no_mcts.select_goals(state)
            actions, _ = hi_agent_no_mcts.select_actions(obs_batch, goals)
            obs_raw, _, _, _ = hi_env_no_mcts.step(actions)
            state = torch.tensor(hi_env_no_mcts.get_global_state(), dtype=torch.float32)
        hi_no_mcts_scores.append(sum(hi_env_no_mcts.compute_final_scores()))
        hi_no_mcts_pops.append(sum(cell.population for row in hi_env_no_mcts.sim.grid for cell in row if cell.tribe))
        hi_no_mcts_foods.append(sum(cell.food for row in hi_env_no_mcts.sim.grid for cell in row if cell.tribe))
        hi_no_mcts_cells.append(sum(count_tribe_cells(hi_env_no_mcts.sim.grid)))
        goals, _ = hi_agent_no_mcts.select_goals(state)
        actions, _ = hi_agent_no_mcts.select_actions(obs_batch, goals)
        for _ in range(10):
            obs_raw, _, _, _ = hi_env_no_mcts.step(actions)
        hi_no_mcts_scores.append(sum(hi_env_no_mcts.compute_final_scores()))
        hi_no_mcts_pops.append(sum(cell.population for row in hi_env_no_mcts.sim.grid for cell in row if cell.tribe))
        hi_no_mcts_foods.append(sum(cell.food for row in hi_env_no_mcts.sim.grid for cell in row if cell.tribe))
        hi_no_mcts_cells.append(sum(count_tribe_cells(hi_env_no_mcts.sim.grid)))

        # === Random Strategy Evaluation ===
        sim = CivilisationSimulation(rows, cols, num_tribes)
        for _ in range(10):
            sim.takeTurn()
        rand_score = 0
        for tribe in range(1, num_tribes + 1):
            territory = sum(cell.tribe == tribe for row in sim.grid for cell in row)
            population = sum(cell.population for row in sim.grid for cell in row if cell.tribe == tribe)
            food = sum(cell.food for row in sim.grid for cell in row if cell.tribe == tribe)
            total_cells = rows * cols
            max_population = total_cells * 10
            if population > 0:
                territory_score = (territory / total_cells)
                population_score = (population / max_population)
                food_score = min((food / (population * 0.2 * 5)), 1.0)
                if food > population * 1.5:
                    overstock_ratio = (food - population * 1.5) / (population * 1.5)
                    penalty = min(overstock_ratio, 1.0)
                    food_score *= (1 - penalty)
                rand_score = (territory_score * 0.4 + population_score * 0.4 + food_score * 0.2) * 100
            else:
                rand_score = 0
        rand_scores.append(rand_score)
        rand_pops.append(sum(cell.population for row in sim.grid for cell in row if cell.tribe))
        
        gc.collect()  #clear memory
        
        if episode % log_interval == 0:
            print(f"Episode {episode}/{num_episodes} evaluated")
        sm = smooth
        if 'axs' not in locals():
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        axs[0, 0].plot(sm(mappo_pops), label="MAPPO", linewidth=2)
        axs[0, 0].plot(sm(qmix_pops), label="QMIX", linewidth=2)
        axs[0, 0].plot(sm(hi_pops), label="Hi-MAPPO", linewidth=2)
        axs[0, 0].plot(sm(hi_no_mcts_pops), label="Hi-MAPPO No MCTS", linewidth=2)
        axs[0, 0].plot(sm(rand_pops), label="Random", linewidth=2)
        axs[0, 0].set_title("Total Population")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].plot(sm(mappo_foods), label="MAPPO", linewidth=2)
        axs[0, 1].plot(sm(qmix_foods), label="QMIX", linewidth=2)
        axs[0, 1].plot(sm(hi_foods), label="Hi-MAPPO", linewidth=2)
        axs[0, 1].plot(sm(hi_no_mcts_foods), label="Hi-MAPPO No MCTS", linewidth=2)
        axs[0, 1].plot(sm(rand_foods), label="Random", linewidth=2)
        axs[0, 1].set_title("Total Food")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        axs[1, 0].plot(sm(mappo_scores), label="MAPPO", linewidth=2)
        axs[1, 0].plot(sm(qmix_scores), label="QMIX", linewidth=2)
        axs[1, 0].plot(sm(hi_scores), label="Hi-MAPPO", linewidth=2)
        axs[1, 0].plot(sm(hi_no_mcts_scores), label="Hi-MAPPO No MCTS", linewidth=2)
        axs[1, 0].plot(sm(rand_scores), label="Random", linewidth=2)
        axs[1, 0].set_title("Final Score")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].plot(sm(mappo_cells), label="MAPPO", linewidth=2)
        axs[1, 1].plot(sm(qmix_cells), label="QMIX", linewidth=2)
        axs[1, 1].plot(sm(hi_cells), label="Hi-MAPPO", linewidth=2)
        axs[1, 1].plot(sm(hi_no_mcts_cells), label="Hi-MAPPO No MCTS", linewidth=2)
        axs[1, 1].plot(sm(rand_cells), label="Random", linewidth=2)
        axs[1, 1].set_title("Occupied Cells")
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 0].grid(True)

if __name__ == "__main__":
    rows, cols = 10, 10
    num_episodes = 1000
    num_tribes = 4
    evaluate_all(rows, cols, num_tribes, num_episodes)

    plt.tight_layout()
    plt.savefig("evaluate_all_model_4plots.png")
    plt.show()
    plt.close()