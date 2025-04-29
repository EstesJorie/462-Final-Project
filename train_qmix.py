import torch
import os
import numpy as np
from civilization_env_qmix import CivilizationEnv_QMIX
from qmix import QMIXAgent
from tqdm import tqdm
import random
import csv

def train_qmix(rows, cols, num_generations, num_tribes, seed=7, log_interval=100, save_dir="trained_models_qmix"):
    """
    Train a QMIX agent in the Civilization environment.

    Args:
        rows, cols: size of the map
        num_generations: total number of training generations
        num_tribes: number of agents (tribes)
        seed: random seed for reproducibility
        log_interval: how often to log scores and render
        save_dir: directory to save trained models

    Training process:
    - Agents interact with environment using ε-greedy exploration
    - Experience replay buffer stores transitions
    - QMIX updates agent nets and mixer via batch training
    - Save models and evaluation scores
    """
    # set random seeds for full reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # initialize environment and agent
    env = CivilizationEnv_QMIX(rows=rows, cols=cols, num_tribes=num_tribes)
    obs_dim = env.rows * env.cols * 3
    state_dim = obs_dim
    act_dim = 3

    agent = QMIXAgent(
        obs_dim=obs_dim,
        state_dim=state_dim,
        act_dim=act_dim,
        n_agents=num_tribes,
        hidden_dim=64,
        mixer_hidden_dim=200,
        buffer_size=10000,
        batch_size=64,
        lr=1e-3,
        gamma=0.99
    )

    # helper to flatten observation
    def preprocess_obs(obs_raw):
        return [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]

    score_log = []
    loss_log = []

    # main training loop
    pbar = tqdm(range(1, num_generations + 1), desc="Training QMIX", unit="gen")
    scores = [0] * num_tribes
    for iteration in pbar:
        obs_raw = env.reset()
        state_raw = torch.tensor(obs_raw.flatten(), dtype=torch.float32)

        for step in range(8):
            obs_batch = preprocess_obs(obs_raw)

            epsilon = max(0.05, 1 - iteration / 2000)  # linear decay of ε

            actions = agent.select_actions(obs_batch, epsilon=epsilon)

            next_obs_raw, rewards, done, _ = env.step(actions)
            next_state_raw = torch.tensor(next_obs_raw.flatten(), dtype=torch.float32)
            next_obs_batch = preprocess_obs(next_obs_raw)

            agent.store_transition(obs_batch, state_raw, actions, rewards, next_obs_batch, next_state_raw)

            obs_raw = next_obs_raw
            state_raw = next_state_raw

        # train agent
        loss = agent.train()
        loss_value = loss.item() if loss is not None else None
        if loss_value is not None:
            loss_log.append((iteration, loss_value))

        # periodic update of target networks
        if iteration % 200 == 0:
            agent.update_targets()

        # log and visualize
        if iteration % log_interval == 0:
            print(f"\n========== Generation {iteration} ==========\n")
            env.render()
            scores = env.compute_final_scores()
            score_log.append((iteration, scores))

        if loss_value is not None:
            print("Loss:", loss_value)

        pbar.set_postfix(loss=loss_value, scores=scores)

    # save models
    os.makedirs(save_dir, exist_ok=True)
    for i, net in enumerate(agent.agent_nets):
        torch.save(net.state_dict(), os.path.join(save_dir, f"qmix_agent_{i}.pth"))
    torch.save(agent.mix_net.state_dict(), os.path.join(save_dir, "qmix_mixer.pth"))
    print(f"✅ QMIX models saved to '{save_dir}/'")

    # save score log
    with open("qmix_score_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation"] + [f"Tribe_{i + 1}_Score" for i in range(num_tribes)])
        for generation, scores in score_log:
            writer.writerow([generation] + scores)

    print("✅ Score logs saved as 'qmix_score_log.csv'.")

if __name__ == "__main__":
    # Example usage
    train_qmix(rows=10, cols=10, num_generations=1000, num_tribes=4)
