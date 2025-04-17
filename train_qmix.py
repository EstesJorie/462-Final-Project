import torch
import os
import numpy as np
from civilization_env_qmix import CivilizationEnv_QMIX  # Environment class (no renaming needed)
from qmix import QMIXAgent                              # QMIX agent definition
from GameController import GameController               # User-configurable controller (custom input)

import random
import numpy as np
import torch

def train_qmix(rows=None, cols=None, generations=None, num_tribes=None, log_interval=100):
    """
    Args:
        rows (int) - Num of grid rows
        cols (int) - Num of grid cols, 
        generations (int) - Num of generations to train
        num_tribes (int) - Num of tribes to train
        log_interval (int) - Interval for logging and rendering
    Returns:
        tuple(agent, env) - Trained QMIX agent and environment
    """
    # === Set random seeds for full reproducibility ===
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():   
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # === Initialize user configuration via GameController ===
    if rows is None or cols is None or generations is None or num_tribes is None:
        controller = GameController()
        rows = rows or controller.getValidDimensions()[0] # Get valid grid dimensions from user
        cols = cols or controller.getValidDimensions()[1] # Get number of generations (training iterations)
        generations = generations or controller.getValidGenerations()
        num_tribes = num_tribes or controller.getValidTribeCount() # Get number of tribes (agents)

    # === Initialize environment and QMIX agent ===
    env = CivilizationEnv_QMIX(rows=rows, cols=cols, num_tribes=num_tribes)

    obs_dim = rows * cols * 3        # Observation: full grid (population, food, tribe) flattened
    state_dim = obs_dim              # In QMIX, global state is often same as full observation
    act_dim = 3                      # Action space: harvest, grow, expand
    agent = QMIXAgent(obs_dim, state_dim, act_dim, n_agents=num_tribes)

    # === Training parameters ===
    total_iterations = generations
    log_interval = 100               # Frequency of rendering/log output

    # === Helper: Preprocess observation into list of tensors for each agent ===
    def preprocess_obs(obs_raw):
        # Each agent receives full observation in this implementation
        return [torch.tensor(obs_raw.flatten(), dtype=torch.float32) for _ in range(num_tribes)]

    # === Main training loop ===
    for iteration in range(1, total_iterations + 1):
        obs_raw = env.reset()                                # Reset environment to start of episode
        state_raw = torch.tensor(obs_raw.flatten(), dtype=torch.float32)  # Flattened global state

        for step in range(10):  # One episode = 10 simulation steps
            obs_batch = preprocess_obs(obs_raw)  # Each agent gets identical full observation
            epsilon = max(0.05, 1 - iteration / 2000)  # Decaying epsilon for exploration
            actions = agent.select_actions(obs_batch, epsilon=epsilon)

            next_obs_raw, rewards, done, _ = env.step(actions)    # Environment step
            next_state_raw = torch.tensor(next_obs_raw.flatten(), dtype=torch.float32)
            next_obs_batch = preprocess_obs(next_obs_raw)

            # Store experience tuple into replay buffer
            agent.store_transition(
                obs_batch, state_raw, actions, rewards, next_obs_batch, next_state_raw
            )

            # Move forward in time
            obs_raw = next_obs_raw
            state_raw = next_state_raw

        # Train from replay buffer
        agent.train()

        # Periodically update target networks
        if iteration % 200 == 0:
            agent.update_targets()

        # Optional visualization/logging
        if iteration % log_interval == 0:
            print(f"\n========== Generation {iteration} ==========")
            env.render()

    # === Save trained model parameters ===
    save_dir = "trained_models_qmix"
    os.makedirs(save_dir, exist_ok=True)

    # Save per-agent networks
    for i, net in enumerate(agent.agent_nets):
        torch.save(net.state_dict(), os.path.join(save_dir, f"qmix_agent_{i}.pth"))

    # Save mixer network
    torch.save(agent.mix_net.state_dict(), os.path.join(save_dir, "qmix_mixer.pth"))

    print("QMIX model has been saved to 'trained_models_qmix/'")
    return agent, env

if __name__ == "__main__":
       trained_agent, final_env = train_qmix(
        rows=5,
        cols=5,  
        generations=1000,  
        num_tribes=3,  
        log_interval=100  
    )
