import torch
import os
import numpy as np
import random
import csv
from tqdm import tqdm
from civilization_env_mappo import CivilizationEnv_MAPPO
from mappo import MAPPOAgent

def train_mappo(rows, cols, generations, num_tribes, seed=7, log_interval=100, save_dir="trained_models_mappo"):
    """
    Train a MAPPO agent in the Civilization environment.

    Args:
        rows, cols: size of the map
        generations: total number of training generations
        num_tribes: number of independent agents (tribes)
        seed: random seed for reproducibility
        log_interval: how often to render and log scores
        save_dir: directory to save trained models

    Training process:
    - Reset environment at each generation
    - Run 15 environment steps to collect trajectories
    - Perform MAPPO updates using collected trajectories
    - Log scores and save models periodically
    """
    # set seeds for full reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # initialize environment and agent
    env = CivilizationEnv_MAPPO(rows=rows, cols=cols, num_tribes=num_tribes)
    obs_dim = env.rows * env.cols * 3
    act_dim = 3
    agent = MAPPOAgent(obs_dim=obs_dim, act_dim=act_dim, num_agents=num_tribes)

    score_log = []

    # main training loop
    pbar = tqdm(range(1, generations + 1), desc="Training MAPPO", unit="gen")
    for iteration in pbar:
        obs_raw = env.reset()
        trajectories = []

        for step in range(8):
            # prepare observations
            obs_batch = []
            for agent_id in range(env.num_agents):
                flat_obs = torch.tensor(obs_raw.flatten(), dtype=torch.float32)
                obs_batch.append(flat_obs)

            # agents take actions
            actions, log_probs = agent.select_action(obs_batch)

            # environment step
            next_obs, rewards, done, _ = env.step(actions)

            # store transition
            trajectories.append({
                'obs': obs_batch,
                'actions': actions,
                'log_probs': log_probs,
                'rewards': rewards
            })

            obs_raw = next_obs

        # update agent using PPO
        loss = agent.update(trajectories)

        # log and visualize progress
        if iteration % log_interval == 0:
            print(f"\n========== Generation {iteration} ==========\n")
            env.render()
            scores = env.compute_final_scores()
            score_log.append((iteration, scores))
            print("Loss:", loss)
            print("Scores:", scores)
            pbar.set_postfix(loss=loss, scores=scores)

    # save trained models
    os.makedirs(save_dir, exist_ok=True)
    for i, actor in enumerate(agent.actors):
        torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_{i}.pth"))
    torch.save(agent.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
    print(f"✅ Models saved to '{save_dir}/'")

    # save score log for future analysis
    score_log_path = "mappo_score_log.csv"
    with open(score_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation"] + [f"Tribe_{i + 1}_Score" for i in range(num_tribes)])
        for generation, scores in score_log:
            writer.writerow([generation] + scores)

    print(f"✅ Score log saved to '{score_log_path}'")

if __name__ == "__main__":
    # Example: train MAPPO on a 10x10 map with 4 tribes for 1000 generations
    train_mappo(rows=10, cols=10, generations=1000, num_tribes=4)
