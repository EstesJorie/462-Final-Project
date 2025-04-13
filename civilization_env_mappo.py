import numpy as np
from civilisation_simulation_mappo import CivilizationSimulation_MAPPO

class CivilizationEnv_MAPPO:
    def __init__(self, rows=5, cols=5, num_tribes=3):
        self.rows = rows
        self.cols = cols
        self.num_tribes = num_tribes
        self.sim = CivilizationSimulation_MAPPO(rows, cols, num_tribes)
        self.num_agents = num_tribes
        self.action_space = [3] * self.num_agents  # 0: gather, 1: grow, 2: expand
        self.observation_space = (rows, cols, 3)

    def reset(self):
        self.sim = CivilizationSimulation_MAPPO(self.rows, self.cols, self.num_tribes)
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs

    def step(self, actions):
        self.sim.take_turn(actions)  # ✅ 使用 MAPPO 动作控制
        obs = self._get_obs()
        rewards = self._compute_rewards()
        done = False
        return obs, rewards, done, {}

    def _compute_rewards(self):
        rewards = [0] * self.num_agents
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    idx = cell.tribe - 1
                    rewards[idx] += cell.population + cell.food
        return rewards

    def render(self):
        self.sim.printGrid()
        self.sim.printDebugInfo()
        self.sim.printStats()
