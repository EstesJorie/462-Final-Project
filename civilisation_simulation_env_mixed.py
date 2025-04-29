import random 
import numpy as np  
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from civilisation_simulation_env import *
import os

class CivilisationSimulationMixed:
    def __init__(self, rows=5, cols=5, num_tribes=3, agents=None):
        """        
        Args:
            rows (int): Number of rows in the grid.
            cols (int): Number of columns in the grid.
            num_tribes (int): Number of tribes (agents).
            agents (list): List of trained agents corresponding to each tribe. 
                            Each element should be a trained agent object (MAPPO, Hi-MAPPO, QMIX).
        """
        self.rows = rows
        self.cols = cols
        self.num_tribes = num_tribes
        self.agents = agents if agents is not None else []

        self.sim = CivilizationSimulation_ENV(rows, cols, num_tribes)
        self.grid = self.sim.grid
        self.last_scores = [0] * num_tribes

    def reset(self):
        """Reset the environment to its initial state."""
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes)
        self.grid = self.sim.grid
        self.last_scores = [0] * self.num_tribes
        gc.collect()  
        return self._get_obs()
    
    def _get_obs(self):
        """Get structured observation of the current grid state.
        
        Returns:
            numpy array [rows x cols x 3] with [pop, food, tribe_id]
        """
        obs = np.zeros((self.rows * self.cols * 3,))  
        index = 0
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                obs[index] = c.population 
                obs[index + 1] = c.food  
                obs[index + 2] = c.tribe if c.tribe else 0  
                index += 3 
        return obs
    
    def step(self, actions=None):
        actions = []
        self.actions_last_step = []  
        obs = self._get_obs()

        for i, agent in enumerate(self.agents):
            agent_obs = torch.tensor(obs.flatten(), dtype=torch.float32)
            action = agent.select_action([agent_obs])[0]  
            actions.append((i + 1, action))  #tuple of (tribe_id, action)
            self.actions_last_step.append((i + 1, action)) 

        self.sim.take_turn(actions)
        self.grid = self.sim.grid
        next_obs = self._get_obs()
        rewards = self._compute_rewards()
        done = False

        return next_obs, rewards, done, {}
        
    def _compute_rewards(self, lambda_score=0.3):
        local_rewards = [0.0] * self.num_tribes

        # Traverse each grid cell to calculate local rewards
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    idx = cell.tribe - 1  # Tribe indices are 1-based, list is 0-based
                    # Compute local reward for this cell:
                    # - 0.5 × population: scaled contribution of population
                    # - 0.05 × food^0.8: diminishing return for food
                    # - +2.0: survival bonus for simply existing
                    local_rewards[idx] += 0.5 * cell.population + 0.05 * (cell.food ** 0.8) + 2.0

        # Get the expansion reward for each tribe (precomputed in environment)
        expansion_bonus = self.sim.expand_reward  # A list of expansion rewards per agent

        # Compute score reward as change in final score from last recorded scores
        current_scores = self.compute_final_scores()
        score_rewards = [curr - prev for curr, prev in zip(current_scores, self.last_scores)]
        self.last_scores = current_scores  # Update last_scores to current for next step

        # Combine local, expansion, and scaled score rewards
        mixed_rewards = [
            local + expand + lambda_score * score
            for local, expand, score in zip(local_rewards, expansion_bonus, score_rewards)
        ]

        return mixed_rewards
    
    # =======================================
    # Compute final civilization score per tribe
    # Score = α × territory + β × population + γ × food efficiency
    # =======================================
    def compute_final_scores(self, H=5, food_per_person=0.2):
        total_cells = self.rows * self.cols
        max_population_per_cell = 10
        scores = []

        # Loop over each tribe to calculate its final score
        for tribe_id in range(1, self.num_tribes + 1):
            territory = 0
            population = 0
            food = 0

            # Aggregate data from all cells belonging to the current tribe
            for i in range(self.rows):
                for j in range(self.cols):
                    cell = self.sim.grid[i][j]
                    if cell.tribe == tribe_id:
                        territory += 1  # Count of cells occupied by the tribe
                        population += cell.population
                        food += cell.food

            # --- Step 1: Raw score components ---
            territory_score = territory / total_cells  # Normalized by total grid area
            population_score = population / (
                        max_population_per_cell * total_cells)  # Normalized by max possible population
            food_needed = population * food_per_person * H  # Total food needed for survival
            food_score = min((food / food_needed), 1.0) if food_needed > 0 else 0  # Cap efficiency at 1.0

            # --- Step 2: Normalization ---
            # Each component is already within [0, 1] range via direct normalization
            norm_territory = territory_score
            norm_population = population_score
            norm_food = food_score

            # --- Step 3: Combine components equally to get final score ---
            final_score = (norm_territory + norm_population + norm_food) / 3.0

            # Scale to percentage format for readability
            scores.append(final_score * 100)

        return scores
    
    def render(self):
        self.sim.printGrid()
        self.sim.printDebugInfo()
        self.sim.printStats()
        print("Final scores:", self.compute_final_scores())

    def renderHeatmap(self, sPath="logs/final_territory_heatmap.png"):
        tribeMap = np.zeros((self.rows, self.cols))

        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    tribeMap[i][j] = cell.tribe
        plt.figure(figsize=(10, 6))
        sns.heatmap(tribeMap, cmap="tab10", cbar=True, linewidths=0.5, linecolor='black', square=True)
        plt.title("Terrority Control")
        os.makedirs(os.path.dirname(sPath), exist_ok=True)

        plt.savefig(sPath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved to {sPath}")
        gc.collect()

    def get_population_score(self):
        population_scores = []
        max_population_per_cell = 10  # assuming this is the max population per cell
        total_max_population = self.rows * self.cols * max_population_per_cell
        
        for tribe_id in range(1, self.num_tribes + 1):
            total_population = sum(self.grid[x][y].population for x in range(self.rows) 
                                for y in range(self.cols) if self.grid[x][y].tribe == tribe_id)
            population_score = total_population / total_max_population if total_max_population > 0 else 0
            population_scores.append(population_score)
        return population_scores
    
    def get_food_score(self):
        food_scores = []
        for tribe_id in range(1, self.num_tribes + 1):
            food_count = sum(1 for x in range(self.rows) for y in range(self.cols) 
                            if self.grid[x][y].tribe == tribe_id and self.grid[x][y].food > 0)
            max_possible_food = self.rows * self.cols  # theoretical maximum
            food_score = food_count / max_possible_food if max_possible_food > 0 else 0
            food_scores.append(food_score)
        return food_scores

    def get_territory_score(self):
        territory_scores = []
        total_cells = self.rows * self.cols
        
        for tribe_id in range(1, self.num_tribes + 1):
            territory_count = sum(1 for x in range(self.rows) for y in range(self.cols) 
                                if self.grid[x][y].tribe == tribe_id)
            territory_score = territory_count / total_cells if total_cells > 0 else 0
            territory_scores.append(territory_score)
        return territory_scores