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

    # =======================================
    # Compute rewards based on local features and score deltas
    # local_rewards: sum of population, food, and survival bonus
    # score_rewards: difference in final score since last step
    # mixed_rewards = local + λ × score_delta
    # =======================================
    def _compute_rewards(self, lambda_score=0.3):
        # Initialize local rewards list for each agent (tribe)
        local_rewards = [0.0] * self.num_agents

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
        for tribe_id in range(1, self.num_agents + 1):
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

    # ======================================================
    # Print grid state, tribe info, and final scores
    # Useful for debugging and visualization
    # ======================================================
    def render(self):
        self.sim.printGrid()
        self.sim.printDebugInfo()
        self.sim.printStats()
        print("Final scores:", self.compute_final_scores())
