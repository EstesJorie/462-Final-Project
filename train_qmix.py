import torch
import os
import numpy as np
from civilization_env_qmix import CivilizationEnv_QMIX
from qmix import QMIXAgent
from tqdm import tqdm
import random
import csv

def train_qmix(rows, cols, num_generations, num_tribes, seed=7, log_interval=100, save_dir="trained_models_qmix"):
    # =======================================
    # Set Random Seed for Reproducibility
    # =======================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # =======================================
    # Environment and Agent Setup
    # =======================================
    env = CivilizationEnv_QMIX(rows=rows, cols=cols, num_tribes=num_tribes)
    obs_dim = 300
    state_dim = obs_dim
    act_dim = 3
    agent = QMIXAgent(
        obs_dim=obs_dim,
        state_dim=state_dim,
        act_dim=act_dim,
        n_agents=num_tribes,
        hidden_dim=64,         
        mixer_hidden_dim=250,  
        buffer_size=10000,
        batch_size=64,
        lr=1e-3,
        gamma=0.99
    )

    # =======================================
    # Helper: Convert raw observation to tensor list
    # Each agent observes the full environment flattened
    # =======================================
    def preprocess_obs(obs_raw):
        return [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]

    score_log = []
    loss_log = []

    # =======================================
    # Training Loop
    # =======================================
    pbar = tqdm(range(1, num_generations + 1), desc="Training QMIX", unit="gen")
    scores = [0] * num_tribes  # Initialize scores
    for iteration in pbar:
        obs_raw = env.reset()
        state_raw = torch.tensor(obs_raw.flatten(), dtype=torch.float32)

        # Run one episode consisting of multiple steps
        for step in range(30):
            obs_batch = preprocess_obs(obs_raw)

            # === ε-greedy exploration ===
            # Epsilon decays linearly to 0.05 over time
            epsilon = max(0.05, 1 - iteration / 2000)

            # === Agents select actions ===
            actions = agent.select_actions(obs_batch, epsilon=epsilon)

            # === Environment executes actions ===
            next_obs_raw, rewards, done, _ = env.step(actions)
            next_state_raw = torch.tensor(next_obs_raw.flatten(), dtype=torch.float32)
            next_obs_batch = preprocess_obs(next_obs_raw)

            # === Store transition in experience replay buffer ===
            agent.store_transition(obs_batch, state_raw, actions, rewards, next_obs_batch, next_state_raw)

            # Update current observation/state
            obs_raw = next_obs_raw
            state_raw = next_state_raw

        # === Train agent using sampled mini-batch from replay buffer ===
        loss = agent.train()
        loss_value = loss.item() if loss is not None else None
        if loss_value is not None:
            loss_log.append((iteration, loss_value))

        # === Periodically update target networks ===
        if iteration % 200 == 0:
            agent.update_targets()

        # === Periodic evaluation and logging ===
        if iteration % log_interval == 0:
            print(f"\n========== Generation {iteration} ==========\n")
            env.render()
            scores = env.compute_final_scores()
            score_log.append((iteration, scores))

        if loss_value is not None:
            print("Loss:", loss_value)

        pbar.set_postfix(loss=loss_value, scores=scores)

    # =======================================
    # Save Trained QMIX Models
    # Save individual agent networks and the mixing network
    # =======================================
    os.makedirs(save_dir, exist_ok=True)
    for i, net in enumerate(agent.agent_nets):
        torch.save(net.state_dict(), os.path.join(save_dir, f"qmix_agent_{i}.pth"))
    torch.save(agent.mix_net.state_dict(), os.path.join(save_dir, "qmix_mixer.pth"))
    print(f"✅ QMIX models saved to '{save_dir}/'")

    # =======================================
    # Save Score Log
    # Save evaluation scores for later analysis or plotting
    # =======================================
    with open("qmix_score_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation"] + [f"Tribe_{i + 1}_Score" for i in range(num_tribes)])
        for generation, scores in score_log:
            writer.writerow([generation] + scores)

    print("✅ Score logs saved as 'qmix_score_log.csv'.")

# =======================================
# Entry Point
# =======================================
if __name__ == "__main__":
    train_qmix(rows=10, cols=10, num_generations=1000, num_tribes=4)
