import torch
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import numpy as np

# Import environment and agent definitions
from civilization_env_mappo import CivilizationEnv_MAPPO
from civilization_env_qmix import CivilizationEnv_QMIX
from civilization_env_hi_mappo import CivilizationEnv_HiMAPPO
from mappo import MAPPOAgent
from qmix import QMIXAgent
from hi_mappo import HiMAPPOAgent
from CivilisationSimulation2 import CivilisationSimulation  # Random strategy simulation

# === Set global seed for reproducibility ===
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# === Helper: Count the number of occupied cells per tribe ===
def count_tribe_cells(grid):
    tribe_cells = {}
    for row in grid:
        for cell in row:
            if cell.tribe:
                tribe_cells[cell.tribe] = tribe_cells.get(cell.tribe, 0) + 1
    return [tribe_cells.get(i + 1, 0) for i in range(3)]

# === Helper: Smoothing function using moving average ===
def smooth(data, window=50):
    return pd.Series(data).rolling(window, min_periods=1).mean()

# === Main evaluation function ===
def evaluate_all(rows, cols, num_tribes, num_episodes=1000, log_interval=100):
    model_dirs = ["trained_models", "trained_models_qmix", "trained_models_hi_mappo"]
    for dir_path in model_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Directory {dir_path} created.")
            return

    obs_dim = rows * cols * 3   # Each cell has 3 features
    act_dim = 3                 # 3 possible actions
    goal_dim = 3                # 3 high-level goals

    # === Load MAPPO agents ===
    mappo_env = CivilizationEnv_MAPPO(rows, cols, num_tribes)
    mappo_agent = MAPPOAgent(obs_dim, act_dim, num_tribes)
    for i, actor in enumerate(mappo_agent.actors):
        actor.load_state_dict(torch.load(os.path.join("trained_models", f"actor_{i}.pth")))
    mappo_agent.critic.load_state_dict(torch.load(os.path.join("trained_models", "critic.pth")))

    # === Load QMIX agents ===
    qmix_env = CivilizationEnv_QMIX(rows, cols, num_tribes)
    qmix_agent = QMIXAgent(obs_dim, obs_dim, act_dim, num_tribes)
    for i, net in enumerate(qmix_agent.agent_nets):
        net.load_state_dict(torch.load(os.path.join("trained_models_qmix", f"qmix_agent_{i}.pth")))
    qmix_agent.mix_net.load_state_dict(torch.load(os.path.join("trained_models_qmix", "qmix_mixer.pth")))

    # === Load Hi-MAPPO agents ===
    hi_env = CivilizationEnv_HiMAPPO(rows, cols, num_tribes)
    hi_agent = HiMAPPOAgent(obs_dim, obs_dim, goal_dim, act_dim, num_tribes)
    for i, worker in enumerate(hi_agent.workers):
        worker.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo", f"worker_{i}.pth")))
    hi_agent.manager.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo", "manager.pth")))

    # === Metrics to track ===
    mappo_pops, qmix_pops, hi_pops, rand_pops = [], [], [], []
    mappo_foods, qmix_foods, hi_foods, rand_foods = [], [], [], []
    mappo_cells, qmix_cells, hi_cells, rand_cells = [], [], [], []

    for episode in range(1, num_episodes + 1):
        # === MAPPO ===
        obs_raw = mappo_env.reset()
        for _ in range(10):
            obs_batch = [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]
            actions, _ = mappo_agent.select_action(obs_batch)
            obs_raw, _, _, _ = mappo_env.step(actions)
        mappo_pops.append(sum(cell.population for row in mappo_env.sim.grid for cell in row if cell.tribe))
        mappo_foods.append(sum(cell.food for row in mappo_env.sim.grid for cell in row if cell.tribe))
        mappo_cells.append(sum(count_tribe_cells(mappo_env.sim.grid)))

        # === QMIX ===
        obs_raw = qmix_env.reset()
        state = torch.tensor(obs_raw.flatten(), dtype=torch.float32)
        for _ in range(10):
            obs_batch = [state.clone().detach().float() for _ in range(num_tribes)]
            actions = qmix_agent.select_actions(obs_batch, epsilon=0.0)
            obs_raw, _, _, _ = qmix_env.step(actions)
            state = torch.tensor(obs_raw.flatten(), dtype=torch.float32)
        qmix_pops.append(sum(cell.population for row in qmix_env.sim.grid for cell in row if cell.tribe))
        qmix_foods.append(sum(cell.food for row in qmix_env.sim.grid for cell in row if cell.tribe))
        qmix_cells.append(sum(count_tribe_cells(qmix_env.sim.grid)))

        # === Hi-MAPPO ===
        obs_raw = hi_env.reset()
        state = torch.tensor(hi_env.get_global_state(), dtype=torch.float32)
        obs_batch = [torch.tensor(o, dtype=torch.float32) for o in hi_env.get_agent_obs()]
        goals, _ = hi_agent.select_goals(state)
        actions, _ = hi_agent.select_actions(obs_batch, goals)
        for _ in range(10):
            obs_raw, _, _, _ = hi_env.step(actions)
        hi_pops.append(sum(cell.population for row in hi_env.sim.grid for cell in row if cell.tribe))
        hi_foods.append(sum(cell.food for row in hi_env.sim.grid for cell in row if cell.tribe))
        hi_cells.append(sum(count_tribe_cells(hi_env.sim.grid)))

        # === Random Strategy ===
        sim = CivilisationSimulation(rows, cols, num_tribes)
        for _ in range(10):
            sim.takeTurn()
        rand_pops.append(sum(cell.population for row in sim.grid for cell in row if cell.tribe))
        rand_foods.append(sum(cell.food for row in sim.grid for cell in row if cell.tribe))
        rand_cells.append(sum(count_tribe_cells(sim.grid)))

        if episode % log_interval == 0:
            print(f"Episode {episode} done.")

    # === Plot smoothed results ===
    sm = lambda x: smooth(x)
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.figure(figsize=(10, 8), dpi=300)

    # --- Left: Population and Food ---
    plt.subplot(1, 2, 1)
    plt.plot(sm(mappo_pops), label="MAPPO Pop", linewidth=2)
    plt.plot(sm(qmix_pops), label="QMIX Pop", linewidth=2)
    plt.plot(sm(hi_pops), label="Hi-MAPPO Pop", linewidth=2)
    plt.plot(sm(rand_pops), label="Random Pop", linewidth=2)
    plt.plot(sm(mappo_foods), label="MAPPO Food", linestyle='--')
    plt.plot(sm(qmix_foods), label="QMIX Food", linestyle='--')
    plt.plot(sm(hi_foods), label="Hi-MAPPO Food", linestyle='--')
    plt.plot(sm(rand_foods), label="Random Food", linestyle='--')
    plt.title("Population & Food Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Amount")
    plt.legend()
    plt.grid(True)

    # --- Right: Expansion (Occupied Cells) ---
    plt.subplot(1, 2, 2)
    plt.plot(sm(mappo_cells), label="MAPPO Cells", linewidth=2)
    plt.plot(sm(qmix_cells), label="QMIX Cells", linewidth=2)
    plt.plot(sm(hi_cells), label="Hi-MAPPO Cells", linewidth=2)
    plt.plot(sm(rand_cells), label="Random Cells", linewidth=2)
    plt.title("Occupied Tribe Cells Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Cell Count")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("evaluate_all_model_smoothed.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                transparent=False,
                edgecolor="none"
                )
    plt.show()

# === Entry Point ===
if __name__ == "__main__":
    rows, cols = map(int, input("Enter no. of rows and columns: ").split())
    num_episodes = int(input("Enter number of generations: "))
    num_tribes = int(input("Enter number of starting tribes: "))
    evaluate_all(rows, cols, num_tribes, num_episodes)
