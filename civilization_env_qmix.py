import numpy as np
from civilisation_simulation_env import CivilizationSimulation_ENV

# Fixed random seed for reproducibility
SEED = 7

# Environment wrapper for QMIX to interact with CivilizationSimulation
class CivilizationEnv_QMIX:
    def __init__(self, rows=5, cols=5, num_tribes=3):
        self.rows = rows                        # Number of grid rows
        self.cols = cols                        # Number of grid columns
        self.num_tribes = num_tribes            # Number of tribes (agents)
        # Create a new simulation instance using the existing ENV-compatible simulator
        self.sim = CivilizationSimulation_ENV(rows, cols, num_tribes, seed=SEED)
        self.num_agents = num_tribes            # Each tribe is treated as one agent
        self.action_space = [3] * self.num_agents  # Each agent has 3 discrete actions: [harvest, grow, expand]
        self.observation_space = (rows, cols, 3)   # Observation shape per step (population, food, tribe ID)

    # Reset the environment and return the initial observation
    def reset(self):
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes)
        return self._get_obs()

    # Construct the full observation of the grid
    def _get_obs(self):
        # Create a 3D numpy array where each cell contains [population, food, tribe_id]
        obs = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                # If the cell is unowned, tribe ID is set to 0
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs

    # Apply a list of actions (one per agent), advance the simulation by one step
    def step(self, actions):
        self.sim.take_turn(actions)         # Execute all agents' actions
        obs = self._get_obs()               # Get new observation after action
        rewards = self._compute_rewards()   # Compute reward for each agent
        done = False                        # The environment never ends in this version
        return obs, rewards, done, {}

    # Reward shaping: reward is a weighted sum of population, food, and a constant bonus
    def _compute_rewards(self):
        rewards = [0] * self.num_agents
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    idx = cell.tribe - 1
                    # Reward for each cell belonging to the agent:
                    # +0.3 per population, +0.2 per food, +2 flat bonus per active cell
                    rewards[idx] += 0.3 * cell.population + 0.2 * cell.food + 2.0
        return rewards

    # Optional: render the simulation to console for visualization and debugging
    def render(self):
        self.sim.printGrid()          # Print simplified grid view
        self.sim.printDebugInfo()     # Print per-cell details
        self.sim.printStats()         # Print per-tribe summary stats
