import torch
import os
import csv
import copy
import random
import numpy as np
from tqdm import tqdm
from hi_mappo import HiMAPPOAgent
from civilization_env_hi_mappo import CivilizationEnv_HiMAPPO

def train_hi_mappo(rows, cols, num_generations, num_tribes, seed=7, log_interval=100, save_dir="trained_models_hi_mappo"):
    """
    Train a Hi-MAPPO agent in the Civilization environment.

    Args:
        rows, cols: size of the map
        num_generations: total number of training generations
        num_tribes: number of independent agents (tribes)
        seed: random seed for reproducibility
        log_interval: how often to log scores and render
        save_dir: directory to save trained models

    Training process:
    - Each generation = reset environment + collect a trajectory
    - Manager selects high-level goals for workers
    - Workers act based on goal-conditioned policy
    - Update manager and workers using collected trajectory
    - Save models and score logs after training
    """
    # set seeds for full reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # initialize environment and agent
    env = CivilizationEnv_HiMAPPO(rows=rows, cols=cols, num_tribes=num_tribes, seed=42)
    obs_dim = env.rows * env.cols * 3
    state_dim = obs_dim
    goal_dim = 3
    act_dim = 3

    agent = HiMAPPOAgent(state_dim, obs_dim, goal_dim, act_dim, num_tribes)
    score_log = []

    # main training loop
    pbar = tqdm(range(1, num_generations + 1), desc="Training HI-MAPPO", unit="gen")
    for gen in pbar:
        env.reset()

        state = torch.tensor(env.get_global_state(), dtype=torch.float32)
        obs_batch = [torch.tensor(o, dtype=torch.float32) for o in env.get_agent_obs()]

        # manager samples goals for each agent
        goal_ids, logp_goal = agent.select_goals(state)
        env.last_scores = goal_ids.tolist()

        # initialize trajectory buffer
        traj = {
            'state': state,
            'obs': [[] for _ in range(num_tribes)],
            'goal': goal_ids,
            'logp_goal': logp_goal,
            'actions': [[] for _ in range(num_tribes)],
            'rewards': [[] for _ in range(num_tribes)]
        }

        # step environment multiple times under fixed goals
        for step in range(8):
            obs_batch = [torch.tensor(o, dtype=torch.float32) for o in env.get_agent_obs()]
            actions, logp_actions = agent.select_actions(obs_batch, goal_ids)
            _, rewards, _, _ = env.step(actions, goals=goal_ids.tolist())

            for i in range(num_tribes):
                traj['obs'][i].append(obs_batch[i])
                traj['actions'][i].append(actions[i])
                traj['rewards'][i].append(rewards[i])

        # update manager and workers
        loss = agent.update([traj])
        loss_value = loss if isinstance(loss, (float, int)) else loss.item()

        # log progress
        if gen % log_interval == 0:
            print(f"\n========== Generation {gen} ==========\n")
            env.render()
            scores = env.compute_final_scores()
            score_log.append((gen, scores))
            print("Loss:", loss_value)
            print("Scores:", scores)
            pbar.set_postfix(loss=loss_value, scores=scores)

    # save models and training logs
    os.makedirs(save_dir, exist_ok=True)

    score_log_path = os.path.join(save_dir, "hi_mappo_score_log.csv")
    with open(score_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation"] + [f"Tribe_{i+1}_Score" for i in range(num_tribes)])
        for generation, scores in score_log:
            writer.writerow([generation] + scores)

    for i, net in enumerate(agent.workers):
        torch.save(net.state_dict(), os.path.join(save_dir, f"worker_{i}.pth"))
    torch.save(agent.manager.state_dict(), os.path.join(save_dir, "manager.pth"))

    print("✅ Hi-MAPPO model and score log saved.")

if __name__ == "__main__":
    # Example usage
    train_hi_mappo(rows=10, cols=10, num_generations=1000, num_tribes=4)
