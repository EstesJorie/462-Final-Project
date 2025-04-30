import os
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from civilization_mixed_env import CivilizationMixedEnv
from mappo import MAPPOAgent
from hi_mappo import HiMAPPOAgent
from qmix import QMIXAgent
from mcts_hi_mappo_compete import MCTS

# set random seed for reproducibility
SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# simulation parameters
rows, cols = 10, 10
num_tribes = 5
num_runs = 5
turns_per_game = 600
steps_per_generation = 8
num_generations = turns_per_game // steps_per_generation

# define agent type for each tribe (1-indexed)
tribe_to_agent_type = {
    1: "MAPPO",
    2: "HiMAPPO",
    3: "HiMAPPO_No_MCTS",
    4: "QMIX",
    5: "Random"
}

obs_dim = rows * cols * 3
act_dim = 3
goal_dim = 3

# load MAPPO agent (only 1 agent used)
mappo_agent = MAPPOAgent(obs_dim, act_dim, num_agents=1)
mappo_agent.actors[0].load_state_dict(torch.load(os.path.join("trained_models_mappo", "actor_0.pth")))
mappo_agent.critic.load_state_dict(torch.load(os.path.join("trained_models_mappo", "critic.pth")))

# load Hi-MAPPO agent
hi_agent = HiMAPPOAgent(obs_dim, obs_dim, goal_dim, act_dim, num_agents=1)
hi_agent.workers[0].load_state_dict(torch.load(os.path.join("trained_models_hi_mappo", "worker_0.pth")))
hi_agent.manager.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo", "manager.pth")))

# load Hi-MAPPO without MCTS
hi_agent_no_mcts = HiMAPPOAgent(obs_dim, obs_dim, goal_dim, act_dim, num_agents=1)
hi_agent_no_mcts.workers[0].load_state_dict(torch.load(os.path.join("trained_models_hi_mappo_no_mcts", "worker_0.pth")))
hi_agent_no_mcts.manager.load_state_dict(torch.load(os.path.join("trained_models_hi_mappo_no_mcts", "manager.pth")))

# load QMIX agent (single-agent version)
qmix_agent = QMIXAgent(
    obs_dim=obs_dim,
    state_dim=obs_dim,
    act_dim=act_dim,
    n_agents=1,
    hidden_dim=64,
    mixer_hidden_dim=200,
    buffer_size=10000,
    batch_size=64,
    lr=1e-3,
    gamma=0.99
)
qmix_agent.agent_nets[0].load_state_dict(torch.load(os.path.join("trained_models_qmix", "qmix_agent_0.pth")))
qmix_agent.mix_net.load_state_dict(torch.load(os.path.join("trained_models_qmix", "qmix_mixer.pth")))

results = []

def compute_individual_scores(env, tribe_id):
    """Compute individual components of score for a tribe."""
    territory, population, food = 0, 0, 0
    for i in range(env.rows):
        for j in range(env.cols):
            cell = env.grid[i][j]
            if cell.tribe == tribe_id:
                territory += 1
                population += cell.population
                food += cell.food

    total_cells = env.rows * env.cols
    max_population_per_cell = 10

    pop_score = population / (max_population_per_cell * total_cells)
    territory_score = territory / total_cells
    food_score = food / (population * 0.2 * 5 + 1e-6)
    food_score = min(food_score, 1.0)

    return pop_score, food_score, territory_score

# start simulation loop
turn_counter = 0
for run_id in range(1, num_runs + 1):
    env = CivilizationMixedEnv(rows, cols, tribe_to_agent_type, seed=run_id + run_id)
    env.reset()
    pbar = tqdm(total=num_generations * steps_per_generation, desc=f"Run {run_id} - Competing", dynamic_ncols=True)

    for generation in range(num_generations):
        for step in range(steps_per_generation):
            turn_counter += 1
            actions = []
            global_state = torch.tensor(env.get_global_state(), dtype=torch.float32)

            # determine actions for each tribe
            for tribe_id in range(1, num_tribes + 1):
                agent_type = tribe_to_agent_type[tribe_id]
                obs = torch.tensor(env.get_obs(tribe_id), dtype=torch.float32)

                if agent_type == "MAPPO":
                    action, _ = mappo_agent.select_action([obs])
                    actions.append(action[0])

                elif agent_type == "HiMAPPO":
                    if generation % 20 == 0:
                        mcts = MCTS(hi_agent, env, controlled_tribe_id=tribe_id, state_tensor=global_state, num_simulations=10)
                        best_goal = mcts.run()
                        goal = torch.tensor([best_goal])
                    else:
                        goal, _ = hi_agent.select_goals(global_state)
                    action, _ = hi_agent.select_actions([obs], goal)
                    actions.append(action[0])

                elif agent_type == "HiMAPPO_No_MCTS":
                    goal, _ = hi_agent_no_mcts.select_goals(global_state)
                    action, _ = hi_agent_no_mcts.select_actions([obs], goal)
                    actions.append(action[0])

                elif agent_type == "QMIX":
                    action = qmix_agent.select_actions([obs], epsilon=0.0)[0]
                    actions.append(action)

                elif agent_type == "Random":
                    actions.append(random.randint(0, 2))

                else:
                    raise ValueError(f"Unknown agent type: {agent_type}")

            env.step(actions)

            scores = env.compute_final_scores()
            for tribe_id in range(1, num_tribes + 1):
                algorithm = tribe_to_agent_type[tribe_id]
                final_score = scores[tribe_id - 1]
                pop_score, food_score, territory_score = compute_individual_scores(env, tribe_id)
                results.append({
                    "run_id": run_id,
                    "algorithm": algorithm,
                    "turn": turn_counter,
                    "pop_score": pop_score,
                    "food_score": food_score,
                    "territory_score": territory_score,
                    "final_score": final_score
                })

            pbar.update(1)

    pbar.close()

# save results to CSV
os.makedirs("logs", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("logs/mixed_agent_results.csv", index=False)
print("[INFO] Score data saved to logs/mixed_agent_results.csv")

# render final map + save territory heatmap
env.render(save_path="logs/final_territory_heatmap.png")

# plot score curve over time
plt.figure(figsize=(10, 6))
for algo in results_df["algorithm"].unique():
    algo_df = results_df[results_df["algorithm"] == algo]
    plt.plot(algo_df["turn"], algo_df["final_score"], label=algo)

plt.xlabel("Turn")
plt.ylabel("Final Score")
plt.title("Agent Compete Final Scores Over Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("logs/score_curves.png")
plt.show()
print("[INFO] Score curve saved to logs/score_curves.png")
