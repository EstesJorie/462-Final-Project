import numpy as np
from civilisation_simulation_env import CivilizationSimulation_ENV

SEED = 7  # Fixed seed for consistent simulation behavior

# ===========================================
# Wrapper Environment for Multi-Agent MAPPO
# Provides standard step/reset API for RL training
# ===========================================
class CivilizationEnv_MAPPO:
    def __init__(self, rows=5, cols=5, num_tribes=3):
        self.rows = rows
        self.cols = cols
        self.num_tribes = num_tribes

        # Initialize the core simulation environment
        self.sim = CivilizationSimulation_ENV(rows, cols, num_tribes, seed=SEED)
        self.grid = self.sim.grid

        self.num_agents = num_tribes                   # One agent per tribe
        self.action_space = [3] * self.num_agents      # Each agent has 3 discrete actions
        self.observation_space = (rows, cols, 3)       # Observation includes population, food, tribe_id
        self.last_scores = [0] * self.num_agents       # For computing score-based reward differences

    # =======================================
    # Reset environment to initial state
    # =======================================
    def reset(self):
        # Create a new simulation instance
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes)
        self.grid = self.sim.grid
        self.last_scores = [0] * self.num_agents
        return self._get_obs()

    # =======================================
    # Convert the simulation grid into structured observations
    # Each cell returns [population, food, tribe_id]
    # =======================================
    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs

    # =======================================
    # Execute actions and return next state and rewards
    # =======================================
    def step(self, actions):
        self.sim.take_turn(actions)   # Simulate one time step
        self.grid = self.sim.grid     # Refresh the local grid reference
        obs = self._get_obs()         # Get updated observations
        rewards = self._compute_rewards()  # Calculate rewards
        done = False                  # No terminal condition defined
        return obs, rewards, done, {}

    # =======================================
    # Compute rewards based on local features and score deltas
    # local_rewards: sum of population, food, and survival bonus
    # score_rewards: difference in final score since last step
    # mixed_rewards = local + λ × score_delta
    # =======================================
    def _compute_rewards(self, lambda_score=0.1):
        local_rewards = [0] * self.num_agents

        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    idx = cell.tribe - 1
                    local_rewards[idx] += 0.3 * cell.population + 0.2 * cell.food + 2.0

        current_scores = self.compute_final_scores()
        score_rewards = [curr - prev for curr, prev in zip(current_scores, self.last_scores)]
        self.last_scores = current_scores

        # Mixed reward signal
        mixed_rewards = [l + lambda_score * s for l, s in zip(local_rewards, score_rewards)]
        return mixed_rewards

    # =======================================
    # Compute final civilization score per tribe
    # Score = α × territory + β × population + γ × food efficiency
    # =======================================
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

            territory_score = (territory / total_cells) * 100
            population_score = (population / (max_population_per_cell * total_cells)) * 100
            food_needed = population * food_per_person * H
            food_score = min((food / food_needed), 1.0) * 100 if food_needed > 0 else 0

            final_score = alpha * territory_score + beta * population_score + gamma * food_score
            scores.append(final_score)

        return scores

    # =======================================
    # Render simulation grid and stats for debugging
    # =======================================
    def render(self):
        self.sim.printGrid()
        self.sim.printDebugInfo()
        self.sim.printStats()
        print("Final scores:", self.compute_final_scores())
