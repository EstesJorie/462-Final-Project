import numpy as np
from civilisation_simulation_env import CivilizationSimulation_ENV

SEED = 7  # Fixed random seed for reproducibility

# ======================================================
# Environment Wrapper for QMIX
# Provides step/reset interface and reward shaping
# for multi-agent training using value decomposition
# ======================================================
class CivilizationEnv_QMIX:
    def __init__(self, rows=5, cols=5, num_tribes=3):
        self.rows = rows
        self.cols = cols
        self.num_tribes = num_tribes

        # Initialize core simulation
        self.sim = CivilizationSimulation_ENV(rows, cols, num_tribes, seed=SEED)

        self.num_agents = num_tribes                     # One agent per tribe
        self.action_space = [3] * self.num_agents        # Each agent has 3 discrete actions
        self.observation_space = (rows, cols, 3)         # [population, food, tribe_id]
        self.last_scores = [0] * self.num_agents         # Used for reward shaping

    # ======================================================
    # Reset environment to initial state
    # ======================================================
    def reset(self):
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes)
        self.last_scores = [0] * self.num_agents
        return self._get_obs()

    # ======================================================
    # Extract grid-based observation from simulation
    # Each cell returns [population, food, tribe_id]
    # ======================================================
    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs

    # ======================================================
    # Perform one simulation step with agent actions
    # Returns: next_obs, reward, done=False, info={}
    # ======================================================
    def step(self, actions):
        self.sim.take_turn(actions)       # Advance environment
        obs = self._get_obs()             # New observation
        rewards = self._compute_rewards() # Get shaped rewards
        done = False                      # No terminal condition
        return obs, rewards, done, {}

    # ======================================================
    # Compute mixed rewards for each tribe/agent
    # Combines local reward (population/food) and score delta
    # ======================================================
    def _compute_rewards(self, lambda_score=0.1):
        local_rewards = [0] * self.num_agents

        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    idx = cell.tribe - 1
                    # Reward components: population, food, and cell occupancy bonus
                    local_rewards[idx] += 0.3 * cell.population + 0.2 * cell.food + 2.0

        current_scores = self.compute_final_scores()  # Get updated score
        score_rewards = [c - l for c, l in zip(current_scores, self.last_scores)]
        self.last_scores = current_scores

        # Mixed reward: local + λ × score_delta
        mixed_rewards = [l + lambda_score * s for l, s in zip(local_rewards, score_rewards)]
        return mixed_rewards

    # ======================================================
    # Compute score based on territory, population, and food
    # Used for reward shaping and monitoring
    # ======================================================
    def compute_final_scores(self, alpha=0.4, beta=0.3, gamma=0.3, H=5, food_per_person=0.2):
        total_cells = self.rows * self.cols
        max_population_per_cell = 10
        scores = []

        for tribe_id in range(1, self.num_agents + 1):
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

            # Score components: normalized to percentage
            territory_score = (territory / total_cells) * 100
            population_score = (population / (max_population_per_cell * total_cells)) * 100
            food_needed = population * food_per_person * H
            food_score = min((food / food_needed), 1.0) * 100 if food_needed > 0 else 0

            # Weighted sum of all three components
            final_score = alpha * territory_score + beta * population_score + gamma * food_score
            scores.append(final_score)

        return scores

    # ======================================================
    # Print grid state, tribe info, and final scores
    # Useful for debugging and visualization
    # ======================================================
    def render(self):
        self.sim.printGrid()
        self.sim.printDebugInfo()
        self.sim.printStats()
        print("Final scores:", self.compute_final_scores())
