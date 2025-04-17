import random
import numpy as np
from civilisation_simulation_env import CivilizationSimulation_ENV

# Fixed seed for reproducibility
SEED = 7

# Environment for hierarchical MAPPO-based civilization simulation
class CivilizationEnv_HiMAPPO:
    def __init__(self, rows=5, cols=5, num_tribes=3, seed=None):
        self.rows = rows                        # Number of rows in the grid
        self.cols = cols                        # Number of columns in the grid
        self.num_tribes = num_tribes            # Number of agents (tribes)
        self.seed = seed                        # Optional custom seed
        # Create a new simulation instance
        self.sim = CivilizationSimulation_ENV(rows, cols, num_tribes)
        # Set random seed if provided (can improve reproducibility for evolution-based methods)
        if seed is not None:
            import random
            random.seed(seed)
        self.num_agents = num_tribes            # One agent per tribe
        self.action_space = [3] * self.num_agents   # Each agent has 3 possible actions
        self.observation_space = (rows, cols, 3)     # Observation shape: [population, food, tribe]

    # Reset the simulation and return the initial observation
    def reset(self):
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes)
        return self._get_obs()

    # Construct the full environment observation
    def _get_obs(self):
        # Observation is a 3D array: grid of cells, each with 3 features
        obs = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                # Store [population, food, tribe ID] for each cell
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs

    # Advance the simulation by one step using agents' actions
    def step(self, actions):
        self.sim.take_turn(actions)            # Apply tribe actions to simulation
        obs = self._get_obs()                  # Observe new state
        rewards = self._compute_rewards()      # Calculate individual rewards
        done = False                           # No terminal condition for now
        return obs, rewards, done, {}

    # Compute per-agent rewards using a shaped reward formula
    def _compute_rewards(self):
        rewards = [0] * self.num_agents
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    idx = cell.tribe - 1
                    # Reward: +0.5 per population, +0.2 per food, +2 flat per controlled cell
                    rewards[idx] += 0.5 * cell.population + 0.2 * cell.food + 2.0
        return rewards

    # Render the current grid state in console output
    def render(self):
        self.sim.printGrid()          # Print tribe layout
        self.sim.printDebugInfo()     # Print detailed per-cell info
        self.sim.printStats()         # Print per-tribe summary stats

    # Return the flattened full-state vector (used for centralized critic)
    def get_global_state(self):
        return self._get_obs().flatten()

    # Return identical local observations for each agent (in this case, all agents see the same full state)
    def get_agent_obs(self):
        state = self._get_obs().flatten()
        return [state.copy() for _ in range(self.num_agents)]
