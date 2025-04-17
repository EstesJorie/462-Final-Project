import torch
import os
from hi_mappo import HiMAPPOAgent                         # Hierarchical MAPPO agent
from civilization_env_hi_mappo import CivilizationEnv_HiMAPPO  # Environment compatible with Hi-MAPPO
import random
import numpy as np
from GameController import GameController           

def train_hi_mappo(rows=None, cols=None, generations=None, num_tribes=None, log_interval=100):
    """
    Args:
        rows (int) - Num of grid rows
        cols (int) - Num of grid cols, 
        generations (int) - Num of generations to train
        num_tribes (int) - Num of tribes to train
        log_interval (int) - Interval for logging and rendering
    Returns:
        tuple(agent, env) - Trained Hi-MAPPO agent and environment
    """
    # === Set global random seed for reproducibility ===
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():   
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # === Get user-defined simulation parameters ===
    if rows is None or cols is None or generations is None or num_tribes is None:
        controller = GameController()
        rows = rows or controller.getValidDimensions()[0] # Get valid grid dimensions from user
        cols = cols or controller.getValidDimensions()[1] # Get number of generations (training iterations)
        generations = generations or controller.getValidGenerations()
        num_tribes = num_tribes or controller.getValidTribeCount() # Get number of tribes (agents)

    # === Initialize the Hi-MAPPO environment ===
    env = CivilizationEnv_HiMAPPO(rows=rows, cols=cols, num_tribes=num_tribes, seed=42)
    obs_dim = rows * cols * 3       # Each cell has 3 values: [population, food, tribe]
    state_dim = obs_dim             # Full global state is flattened grid
    goal_dim = 3                    # Number of high-level goals (e.g., expand, grow, harvest)
    act_dim = 3                     # Number of low-level actions (same as environment)
    agent = HiMAPPOAgent(state_dim, obs_dim, goal_dim, act_dim, num_tribes)

    log_interval = 100              # How often to render the environment

    # === Main training loop ===
    for gen in range(1, generations + 1):
        # Reset environment and get initial observations
        obs_raw = env.reset()
        state = torch.tensor(env.get_global_state(), dtype=torch.float32)           # Global state
        obs_batch = [torch.tensor(o, dtype=torch.float32) for o in env.get_agent_obs()]  # Per-agent observation

        # === High-level: sample goals for each agent ===
        goal_ids, logp_goal = agent.select_goals(state)                             # Sample one goal per agent

        # === Low-level: select actions based on goal and observation ===
        actions, logp_actions = agent.select_actions(obs_batch, goal_ids)

        # === Prepare trajectory data for training ===
        traj = {
            'state': state,                 # Global state
            'obs': obs_batch,              # Per-agent observations
            'goal': goal_ids,              # Assigned goals for this generation
            'logp_goal': logp_goal,        # Log-probability of goals (for PPO)
            'actions': actions,            # Actions taken by workers
            'rewards': []                  # Accumulated reward list
        }

        # === Step through simulation ===
        for step in range(10):  # Each generation runs for 10 simulation steps
            next_obs_raw, rewards, done, _ = env.step(actions)      # Environment step
            traj['rewards'].append(sum(rewards))                    # Store total reward (can be agent-specific if needed)

        # === Update high-level and low-level networks ===
        agent.update([traj])  # Here we pass a list of one trajectory (can be batched in future)

        # === Optional logging/visualization ===
        if gen % log_interval == 0:
            print(f"\n===== Generation {gen} =====")
            env.render()

    # === Save trained models to disk ===
    os.makedirs("trained_models_hi_mappo", exist_ok=True)
    for i, net in enumerate(agent.workers):
        torch.save(net.state_dict(), f"trained_models_hi_mappo/worker_{i}.pth")     # Save each worker network
    torch.save(agent.manager.state_dict(), "trained_models_hi_mappo/manager.pth")   # Save manager network

    print("✅ Hi-MAPPO model has been saved to trained_models_hi_mappo/")
    return agent, env

if __name__ == "__main__":
    trained_agent, final_env = train_hi_mappo(
        rows=5,
        cols=5,  
        generations=1000,  
        num_tribes=3,  
        log_interval=100  
    )