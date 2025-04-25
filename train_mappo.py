import torch
import os
import numpy as np
import random
import csv
from tqdm import tqdm
from civilization_env_mappo import CivilizationEnv_MAPPO
from mappo import MAPPOAgent

def train_mappo(rows, cols, generations, num_tribes, seed=7, log_interval=100, save_dir="trained_models_mappo"):
    # ======================================
    # Set deterministic behavior for reproducibility
    # ======================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ======================================
    # Initialize environment and MAPPO agent
    # ======================================
    env = CivilizationEnv_MAPPO(rows=rows, cols=cols, num_tribes=num_tribes)
    obs_dim = env.rows * env.cols * 3
    act_dim = 3
    agent = MAPPOAgent(obs_dim=obs_dim, act_dim=act_dim, num_agents=num_tribes)

    score_log = []

    # ======================================
    # Training Loop
    # ======================================
    pbar = tqdm(range(1, generations + 1), desc="Training MAPPO", unit="gen")
    for iteration in pbar:
        obs_raw = env.reset()
        trajectories = []

        for step in range(15):
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

        # Update agent
        loss = agent.update(trajectories)

        # Logging
        if iteration % log_interval == 0:
            print(f"\n========== Generation {iteration} ==========")
            env.render()
            scores = env.compute_final_scores()
            score_log.append((iteration, scores))
            print("Loss:", loss)
            print("Scores:", scores)
            pbar.set_postfix(loss=loss, scores=scores)

    # ======================================
    # Save Trained Models
    # ======================================
    os.makedirs(save_dir, exist_ok=True)
    for i, actor in enumerate(agent.actors):
        torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_{i}.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
    print(f"✅ Models saved to '{save_dir}/'")

    # ======================================
    # Save Score Log
    # ======================================
    score_log_path = "mappo_score_log.csv"
    with open(score_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation"] + [f"Tribe_{i + 1}_Score" for i in range(num_tribes)])
        for generation, scores in score_log:
            writer.writerow([generation] + scores)

    print(f"✅ Score log saved to '{score_log_path}'")

if __name__ == "__main__":
    # Example usage
    train_mappo(rows=10, cols=10, generations=1000, num_tribes=4)