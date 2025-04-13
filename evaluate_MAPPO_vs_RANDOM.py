
import torch
import random
import matplotlib.pyplot as plt
import os
import pandas as pd
from civilization_env_mappo import CivilizationEnv_MAPPO
from mappo import MAPPOAgent
from civilisation_simulation_2 import CivilizationSimulation
# Count the number of grid cells occupied by each tribe
def count_tribe_cells(grid):
    tribe_cells = {}
    for row in grid:
        for cell in row:
            if cell.tribe:  # If this cell is occupied by a tribe
                tribe_cells[cell.tribe] = tribe_cells.get(cell.tribe, 0) + 1
    # Return a list of cell counts for each tribe (1 to 3)
    return [tribe_cells.get(i + 1, 0) for i in range(3)]

# Apply smoothing (rolling average) to a list of data
def smooth(data, window=50):
    return pd.Series(data).rolling(window, min_periods=1).mean()

# Run MAPPO and Random policies in the environment and compare their performance
def evaluate_all(num_episodes=1000, log_interval=100, model_path="trained_models"):
    # Environment and agent configuration
    rows, cols, num_tribes = 8, 8, 3
    obs_dim = rows * cols * 3  # Observation space dimension
    act_dim = 3   # Action space dimension

    # === Initialize MAPPO Agent ===
    mappo_env = CivilizationEnv_MAPPO(rows, cols, num_tribes)
    agent = MAPPOAgent(obs_dim, act_dim, num_tribes)
    # Load pretrained actor and critic models
    for i, actor in enumerate(agent.actors):
        actor.load_state_dict(torch.load(os.path.join(model_path, f"actor_{i}.pth")))
    agent.critic.load_state_dict(torch.load(os.path.join(model_path, "critic.pth")))

    # === Prepare data containers for both MAPPO and Random results ===
    mappo_pops, mappo_foods, mappo_cells = [], [], []
    rand_pops, rand_foods, rand_cells = [], [], []

    # === Main Evaluation Loop ===
    for episode in range(1, num_episodes + 1):
        # --- Evaluate MAPPO agent ---
        obs_raw = mappo_env.reset()  # Reset environment
        for _ in range(10):   # Run fixed number of steps
            # Create batched observations for each agent
            obs_batch = [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]
            actions, _ = agent.select_action(obs_batch)   # Select actions using MAPPO
            obs_raw, _, _, _ = mappo_env.step(actions)   # Step environment

        # Collect statistics from MAPPO-controlled simulation
        pop_sum, food_sum = 0, 0
        for row in mappo_env.sim.grid:
            for cell in row:
                if cell.tribe:
                    pop_sum += cell.population
                    food_sum += cell.food
        mappo_pops.append(pop_sum)
        mappo_foods.append(food_sum)
        mappo_cells.append(sum(count_tribe_cells(mappo_env.sim.grid)))

        # --- Evaluate Random agent ---
        sim = CivilizationSimulation(rows, cols, num_tribes)
        for _ in range(10):    # Run simulation with random actions
            sim.take_turn()

        # Collect statistics from randomly controlled simulation
        pop_sum, food_sum = 0, 0
        for row in sim.grid:
            for cell in row:
                if cell.tribe:
                    pop_sum += cell.population
                    food_sum += cell.food
        rand_pops.append(pop_sum)
        rand_foods.append(food_sum)
        rand_cells.append(sum(count_tribe_cells(sim.grid)))

        # Print progress
        if episode % log_interval == 0:
            print(f"Episode {episode} done.")

    # === Smooth data for better visualization ===
    sm_mappo_pops = smooth(mappo_pops)
    sm_rand_pops = smooth(rand_pops)
    sm_mappo_foods = smooth(mappo_foods)
    sm_rand_foods = smooth(rand_foods)
    sm_mappo_cells = smooth(mappo_cells)
    sm_rand_cells = smooth(rand_cells)

    # === Plot results ===
    plt.figure(figsize=(12, 6))

    # Plot population and food over episodes
    plt.subplot(1, 2, 1)
    plt.plot(sm_mappo_pops, label="MAPPO Population", linewidth=2)
    plt.plot(sm_rand_pops, label="Random Population", linewidth=2)
    plt.plot(sm_mappo_foods, label="MAPPO Food", linestyle="--", linewidth=1.5)
    plt.plot(sm_rand_foods, label="Random Food", linestyle="--", linewidth=1.5)
    plt.title("Population & Food Comparison (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Amount")
    plt.legend()
    plt.grid(True)

    # Plot tribe cell occupancy over episodes
    plt.subplot(1, 2, 2)
    plt.plot(sm_mappo_cells, label="MAPPO Tribe Cells", linewidth=2)
    plt.plot(sm_rand_cells, label="Random Tribe Cells", linewidth=2)
    plt.title("Occupied Tribe Cells Over Time (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Cell Count")
    plt.legend()
    plt.grid(True)

    # Final layout adjustments and save
    plt.tight_layout()
    plt.savefig("mappo_vs_random_smoothed.png")
    plt.show()

# Run the evaluation function if this script is executed directly
if __name__ == "__main__":
    evaluate_all()
