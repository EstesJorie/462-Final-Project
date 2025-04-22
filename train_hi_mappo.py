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
    # ========================================================
    # Set Seed for Reproducibility
    # ========================================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ========================================================
    # Initialize Environment and Agent
    # ========================================================
    env = CivilizationEnv_HiMAPPO(rows=rows, cols=cols, num_tribes=num_tribes, seed=42)
    obs_dim = 300
    state_dim = obs_dim
    goal_dim = 3
    act_dim = 3

    agent = HiMAPPOAgent(state_dim, obs_dim, goal_dim, act_dim, num_tribes)
    score_log = []

    # ========================================================
    # Training Loop
    # ========================================================
    pbar = tqdm(range(1, num_generations + 1), desc="Training HI-MAPPO", unit="gen")
    for gen in pbar:
        obs_raw = env.reset()
        state = torch.tensor(env.get_global_state(), dtype=torch.float32)
        obs_batch = [torch.tensor(o, dtype=torch.float32) for o in env.get_agent_obs()]

        use_mcts = gen <= 500
        goal_ids, logp_goal = agent.select_goals(state, env=copy.deepcopy(env), use_mcts=use_mcts)
        actions, logp_actions = agent.select_actions(obs_batch, goal_ids)

        env.last_scores = goal_ids.tolist()

        traj = {
            'state': state,
            'obs': obs_batch,
            'goal': goal_ids,
            'logp_goal': logp_goal,
            'actions': actions,
            'rewards': []
        }

        for step in range(10):
            next_obs_raw, rewards, done, _ = env.step(actions)
            traj['rewards'].append(sum(rewards))

        loss = agent.update([traj])
        loss_value = loss if isinstance(loss, (float, int)) else loss.item()

        if gen % log_interval == 0:
            print(f"\n===== Generation {gen} =====")
            env.render()
            scores = env.compute_final_scores()
            score_log.append((gen, scores))
            print("Loss:", loss_value)
            print("Scores:", scores)
            pbar.set_postfix(loss=loss_value, scores=scores)

    # ========================================================
    # Save Results
    # ========================================================
    os.makedirs(save_dir, exist_ok=True)

    # Save Score Log
    score_log_path = os.path.join(save_dir, "hi_mappo_score_log.csv")
    with open(score_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation"] + [f"Tribe_{i+1}_Score" for i in range(num_tribes)])
        for generation, scores in score_log:
            writer.writerow([generation] + scores)

    # Save Model Checkpoints
    for i, net in enumerate(agent.workers):
        torch.save(net.state_dict(), os.path.join(save_dir, f"worker_{i}.pth"))
    torch.save(agent.manager.state_dict(), os.path.join(save_dir, "manager.pth"))

    print("✅ Hi-MAPPO model and score log saved.")

if __name__ == "__main__":
    # Example usage
    train_hi_mappo(rows=10, cols=10, num_generations=1000, num_tribes=4)