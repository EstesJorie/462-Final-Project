import numpy as np
import random
import os
from civilisation_simulation_env import CivilizationSimulation_ENV

class CivilizationMixedEnv:
    """
    Civilization environment where different tribes can be controlled
    by different types of agents (e.g., MAPPO, HiMAPPO, QMIX, Random).

    Supports flexible agent competition in the same simulation.
    """

    def __init__(self, rows=10, cols=10, tribe_to_agent_type=None, seed=None):
        self.rows = rows
        self.cols = cols
        self.num_tribes = len(tribe_to_agent_type)
        self.tribe_to_agent_type = tribe_to_agent_type
        self.seed = seed

        self.sim = CivilizationSimulation_ENV(rows, cols, self.num_tribes, seed=seed)
        self.grid = self.sim.grid

        self._boost_initial_conditions()

    def reset(self):
        """Reset environment to initial state."""
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes, seed=self.seed)
        self.grid = self.sim.grid
        self._boost_initial_conditions()
        return self.get_all_obs()

    def _boost_initial_conditions(self):
        """boost initial food and population to prevent early extinction."""
        for row in self.grid:
            for cell in row:
                if cell.population > 0:
                    cell.food = max(cell.food, 30.0)
                    cell.population = max(cell.population, 3)

    def step(self, tribe_actions):
        """Apply tribe actions and update environment."""
        self.sim.take_turn(tribe_actions)
        self.grid = self.sim.grid
        self._emergency_food_support()

    def _emergency_food_support(self):
        """provide minimum food after each step to prevent starvation collapse."""
        for row in self.grid:
            for cell in row:
                if cell.population > 0 and cell.food < 5:
                    cell.food += 5.0

    def get_obs(self, tribe_id):
        """Return flattened observation for a specific tribe."""
        obs = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs.flatten()

    def get_all_obs(self):
        """Return list of observations for all tribes."""
        return [self.get_obs(tid) for tid in range(1, self.num_tribes + 1)]

    def get_global_state(self):
        """Return global flattened observation (identical for all agents)."""
        return self.get_obs(1)

    def compute_final_scores(self, H=5, food_per_person=0.2):
        """
        Compute final civilization scores for each tribe.

        Based on:
        - Territory controlled
        - Population size
        - Food sufficiency over H turns
        """
        total_cells = self.rows * self.cols
        max_population_per_cell = 10
        scores = []

        for tribe_id in range(1, self.num_tribes + 1):
            territory = 0
            population = 0
            food = 0

            for i in range(self.rows):
                for j in range(self.cols):
                    cell = self.sim.grid[i][j]
                    if cell.tribe == tribe_id:
                        territory += 1
                        population += cell.population
                        food += cell.food

            territory_score = territory / total_cells
            population_score = population / (max_population_per_cell * total_cells)

            food_needed = population * food_per_person * H
            food_ratio = food / (food_needed + 1e-6)

            if food_ratio >= 1.0:
                food_score = 1.0 - 0.1 * (food_ratio - 1.0)
                food_score = max(food_score, 0.0)
            else:
                food_score = food_ratio

            final_score = (
                0.4 * territory_score +
                0.4 * population_score +
                0.2 * food_score
            )
            scores.append(final_score * 100)

        return scores

    def render(self, save_path=None):
        """Print civilization grid and optionally save final territory heatmap."""
        import matplotlib.pyplot as plt

        print("\n=== Civilization Map ===")
        for row in self.grid:
            line = ""
            for cell in row:
                if cell.population > 0:
                    line += f"T{cell.tribe:<3}"
                else:
                    line += ".   "
            print(line)

        if save_path:
            territory_map = np.zeros((self.rows, self.cols))
            for i in range(self.rows):
                for j in range(self.cols):
                    c = self.grid[i][j]
                    territory_map[i][j] = c.tribe if c.tribe else 0

            plt.figure(figsize=(8, 6))
            plt.title("Final Territory Heatmap")
            plt.imshow(territory_map, cmap='tab20', interpolation='nearest')
            plt.colorbar()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            print(f"[INFO] Final heatmap saved to {save_path}")
