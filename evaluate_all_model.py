import torch
import random
import matplotlib.pyplot as plt
import os
import gc
import pandas as pd
import numpy as np
from tqdm import tqdm

# === Custom Environments and Agents ===
from civilization_env_mappo import CivilizationEnv_MAPPO
from civilization_env_qmix import CivilizationEnv_QMIX
from civilization_env_hi_mappo import CivilizationEnv_HiMAPPO
from civilization_env_hi_mappo_no_mcts import CivilizationEnv_HiMAPPO as CivilizationEnv_HiMAPPO_No_MCTS
from mappo import MAPPOAgent
from qmix import QMIXAgent
from hi_mappo import HiMAPPOAgent
from CivilisationSimulation2 import CivilisationSimulation

# set seed for reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def count_tribe_cells(grid):
    """Count number of occupied cells per tribe."""
    tribe_cells = {}
    for row in grid:
        for cell in row:
            if cell.tribe:
                tribe_cells[cell.tribe] = tribe_cells.get(cell.tribe, 0) + 1
    return [tribe_cells.get(i + 1, 0) for i in range(3)]

def smooth(data, window=50):
    """Simple moving average smoothing."""
    return pd.Series(data).rolling(window, min_periods=1).mean()

def evaluate_all(rows, cols, num_tribes, num_episodes=1000, log_interval=100, save_interval=500):
    """
    Evaluate MAPPO, QMIX, Hi-MAPPO, Hi-MAPPO (no MCTS) and Random baseline.

    - Load trained models
    - Run multiple evaluation episodes
    - Track population, food, cells, final scores
    - Save periodic checkpoints and final plot
    """
    obs_dim = rows * cols * 3
    act_dim = 3
    goal_dim = 3
    steps_per_episode = 8

    # load trained models
    mappo_env = CivilizationEnv_MAPPO(rows, cols, num_tribes)
    mappo_agent = MAPPOAgent(obs_dim, act_dim, num_tribes)
    for i, actor in enumerate(mappo_agent.actors):
        actor.load_state_dict(torch.load(os.path.join("trained_models_mappo", f"actor_{i}.pth")))
    mappo_agent.critic.load_state_dict(torch.load(os.path.join("trained_models_mappo", "critic.pth")))

    qmix_env = CivilizationEnv_QMIX(rows, cols, num_tribes)
    qmix_agent = QMIXAgent(
        obs_dim=obs_dim, state_dim=obs_dim, act_dim=act_dim, n_agents=num_tribes,
        hidden_dim=64, mixer_hidden_dim=200, buffer_size=10000, batch_size=64, lr=1e-3, gamma=0.99
    )
    for i, net in enumerate(qmix_agent.agent_nets):
        net.load_state_dict(torch.load(os.path.join("trained_models_qmix", f"qmix_agent_{i}.pth")))
    qmix_agent.mix_net.load_state_dict(torch.load(os.path.join("trained_models_qmix", "qmix_mixer.pth")))

    hi_env = CivilizationEnv_HiMAPPO(rows, cols, num_tribes)
    hi_agent = HiMAPPOAgent(obs_dim, obs_dim, goal_dim, act_dim, num_tribes)
    for i, worker in enumerate(hi_agent.workers):
        worker.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo", f"worker_{i}.pth")))
    hi_agent.manager.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo", "manager.pth")))

    hi_env_no_mcts = CivilizationEnv_HiMAPPO_No_MCTS(rows, cols, num_tribes)
    hi_agent_no_mcts = HiMAPPOAgent(obs_dim, obs_dim, goal_dim, act_dim, num_tribes)
    for i, worker in enumerate(hi_agent_no_mcts.workers):
        worker.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo_no_mcts", f"worker_{i}.pth")))
    hi_agent_no_mcts.manager.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo_no_mcts", "manager.pth")))

    # initialize logs
    logs = {key: [] for key in [
        'mappo_pops', 'qmix_pops', 'hi_pops', 'rand_pops', 'hi_no_mcts_pops',
        'mappo_foods', 'qmix_foods', 'hi_foods', 'rand_foods', 'hi_no_mcts_foods',
        'mappo_cells', 'qmix_cells', 'hi_cells', 'rand_cells', 'hi_no_mcts_cells',
        'mappo_scores', 'qmix_scores', 'hi_scores', 'rand_scores', 'hi_no_mcts_scores'
    ]}

    pbar = tqdm(total=num_episodes, desc="Evaluating", dynamic_ncols=True)

    for episode in range(1, num_episodes + 1):
        # MAPPO
        obs_raw = mappo_env.reset()
        for _ in range(steps_per_episode):
            obs_batch = [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]
            actions, _ = mappo_agent.select_action(obs_batch)
            obs_raw, _, _, _ = mappo_env.step(actions)
        logs['mappo_scores'].append(sum(mappo_env.compute_final_scores()))
        logs['mappo_pops'].append(sum(cell.population for row in mappo_env.sim.grid for cell in row if cell.tribe))
        logs['mappo_foods'].append(sum(cell.food for row in mappo_env.sim.grid for cell in row if cell.tribe))
        logs['mappo_cells'].append(sum(count_tribe_cells(mappo_env.sim.grid)))

        # QMIX
        obs_raw = qmix_env.reset()
        for _ in range(steps_per_episode):
            obs_batch = [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]
            actions = qmix_agent.select_actions(obs_batch, epsilon=0.0)
            obs_raw, _, _, _ = qmix_env.step(actions)
        logs['qmix_scores'].append(sum(qmix_env.compute_final_scores()))
        logs['qmix_pops'].append(sum(cell.population for row in qmix_env.sim.grid for cell in row if cell.tribe))
        logs['qmix_foods'].append(sum(cell.food for row in qmix_env.sim.grid for cell in row if cell.tribe))
        logs['qmix_cells'].append(sum(count_tribe_cells(qmix_env.sim.grid)))

        # Hi-MAPPO
        obs_raw = hi_env.reset()
        state = torch.tensor(hi_env.get_global_state(), dtype=torch.float32)
        for _ in range(steps_per_episode):
            obs_batch = [torch.tensor(o, dtype=torch.float32) for o in hi_env.get_agent_obs()]
            goals, _ = hi_agent.select_goals(state)
            actions, _ = hi_agent.select_actions(obs_batch, goals)
            obs_raw, _, _, _ = hi_env.step(actions)
            state = torch.tensor(hi_env.get_global_state(), dtype=torch.float32)
        logs['hi_scores'].append(sum(hi_env.compute_final_scores()))
        logs['hi_pops'].append(sum(cell.population for row in hi_env.sim.grid for cell in row if cell.tribe))
        logs['hi_foods'].append(sum(cell.food for row in hi_env.sim.grid for cell in row if cell.tribe))
        logs['hi_cells'].append(sum(count_tribe_cells(hi_env.sim.grid)))

        # Hi-MAPPO No MCTS
        obs_raw = hi_env_no_mcts.reset()
        state = torch.tensor(hi_env_no_mcts.get_global_state(), dtype=torch.float32)
        for _ in range(steps_per_episode):
            obs_batch = [torch.tensor(o, dtype=torch.float32) for o in hi_env_no_mcts.get_agent_obs()]
            goals, _ = hi_agent_no_mcts.select_goals(state)
            actions, _ = hi_agent_no_mcts.select_actions(obs_batch, goals)
            obs_raw, _, _, _ = hi_env_no_mcts.step(actions)
            state = torch.tensor(hi_env_no_mcts.get_global_state(), dtype=torch.float32)
        logs['hi_no_mcts_scores'].append(sum(hi_env_no_mcts.compute_final_scores()))
        logs['hi_no_mcts_pops'].append(sum(cell.population for row in hi_env_no_mcts.sim.grid for cell in row if cell.tribe))
        logs['hi_no_mcts_foods'].append(sum(cell.food for row in hi_env_no_mcts.sim.grid for cell in row if cell.tribe))
        logs['hi_no_mcts_cells'].append(sum(count_tribe_cells(hi_env_no_mcts.sim.grid)))

        # Random baseline
        sim = CivilisationSimulation(rows, cols, num_tribes)
        for _ in range(steps_per_episode):
            sim.takeTurn()
        rand_score = 0
        for tribe in range(1, num_tribes + 1):
            territory = sum(cell.tribe == tribe for row in sim.grid for cell in row)
            population = sum(cell.population for row in sim.grid for cell in row if cell.tribe == tribe)
            food = sum(cell.food for row in sim.grid for cell in row if cell.tribe == tribe)
            if population > 0:
                territory_score = territory / (rows * cols)
                population_score = population / (rows * cols * 10)
                food_score = min(food / (population * 0.2 * 5), 1.0)
                rand_score += (territory_score * 0.4 + population_score * 0.4 + food_score * 0.2) * 100
        logs['rand_scores'].append(rand_score)
        logs['rand_pops'].append(sum(cell.population for row in sim.grid for cell in row if cell.tribe))
        logs['rand_foods'].append(sum(cell.food for row in sim.grid for cell in row if cell.tribe))
        logs['rand_cells'].append(sum(count_tribe_cells(sim.grid)))

        pbar.update(1)

        if episode % save_interval == 0 or episode == num_episodes:
            pd.DataFrame(logs).to_csv(f"evaluate_all_checkpoint_{episode}.csv", index=False)

        gc.collect()

    pbar.close()

    # plot and save
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    sm = smooth

    axs[0, 0].plot(sm(logs['mappo_pops']), label="MAPPO")
    axs[0, 0].plot(sm(logs['qmix_pops']), label="QMIX")
    axs[0, 0].plot(sm(logs['hi_pops']), label="Hi-MAPPO")
    axs[0, 0].plot(sm(logs['hi_no_mcts_pops']), label="Hi-MAPPO No MCTS")
    axs[0, 0].plot(sm(logs['rand_pops']), label="Random")
    axs[0, 0].set_title("Total Population")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].plot(sm(logs['mappo_foods']), label="MAPPO")
    axs[0, 1].plot(sm(logs['qmix_foods']), label="QMIX")
    axs[0, 1].plot(sm(logs['hi_foods']), label="Hi-MAPPO")
    axs[0, 1].plot(sm(logs['hi_no_mcts_foods']), label="Hi-MAPPO No MCTS")
    axs[0, 1].plot(sm(logs['rand_foods']), label="Random")
    axs[0, 1].set_title("Total Food")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].plot(sm(logs['mappo_scores']), label="MAPPO")
    axs[1, 0].plot(sm(logs['qmix_scores']), label="QMIX")
    axs[1, 0].plot(sm(logs['hi_scores']), label="Hi-MAPPO")
    axs[1, 0].plot(sm(logs['hi_no_mcts_scores']), label="Hi-MAPPO No MCTS")
    axs[1, 0].plot(sm(logs['rand_scores']), label="Random")
    axs[1, 0].set_title("Final Score")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].plot(sm(logs['mappo_cells']), label="MAPPO")
    axs[1, 1].plot(sm(logs['qmix_cells']), label="QMIX")
    axs[1, 1].plot(sm(logs['hi_cells']), label="Hi-MAPPO")
    axs[1, 1].plot(sm(logs['hi_no_mcts_cells']), label="Hi-MAPPO No MCTS")
    axs[1, 1].plot(sm(logs['rand_cells']), label="Random")
    axs[1, 1].set_title("Occupied Cells")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig("evaluate_all_model_4plots.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    rows, cols = 10, 10
    num_episodes = 1000
    num_tribes = 5
    evaluate_all(rows, cols, num_tribes, num_episodes)
