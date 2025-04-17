import numpy as np
from civilisation_simulation_env import CivilizationSimulation_ENV

SEED = 7
# === Environment wrapper for CivilizationSimulation, adapted for MAPPO ===
class CivilizationEnv_MAPPO:
    def __init__(self, rows=5, cols=5, num_tribes=3):
        self.rows = rows
        self.cols = cols
        self.num_tribes = num_tribes
        self.sim = CivilizationSimulation_ENV(rows, cols, num_tribes, seed=SEED)  # Core simulation engine
        self.num_agents = num_tribes
        self.action_space = [3] * self.num_agents  # Each agent has 3 discrete actions: 0=gather, 1=grow, 2=expand
        self.observation_space = (rows, cols, 3)   # Observation: for each cell - [population, food, tribe_id]

    # === Reset environment to a new episode ===
    def reset(self):
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes)
        return self._get_obs()

    # === Convert grid state into structured observation (used by agents) ===
    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 3))  # Initialize observation tensor
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs

    # === Execute one environment step using provided actions from MAPPO agents ===
    def step(self, actions):
        self.sim.take_turn(actions)  # Step forward using the list of actions per agent
        obs = self._get_obs()
        rewards = self._compute_rewards()  # Calculate per-agent rewards based on new state
        done = False  # This environment runs indefinitely; define terminal condition externally if needed
        return obs, rewards, done, {}

    # === Calculate rewards for each tribe based on population + food across their cells ===
    def _compute_rewards(self):
        rewards = [0] * self.num_agents
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    idx = cell.tribe - 1
                    rewards[idx] += 0.3 * cell.population + 0.2 * cell.food + 2.0
        return rewards

    # === Render the current environment state (calls sim’s display methods) ===
    def render(self):
        self.sim.printGrid()
        self.sim.printDebugInfo()
        self.sim.printStats()
