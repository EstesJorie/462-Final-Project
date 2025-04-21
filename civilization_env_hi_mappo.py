import random
import numpy as np
from civilisation_simulation_env import CivilizationSimulation_ENV

SEED = 7  # Fixed seed for reproducibility

# =====================================================
# Hi-MAPPO Environment Wrapper
# Custom environment wrapper used for hierarchical RL
# Supports both global and agent-level observations
# =====================================================
class CivilizationEnv_HiMAPPO:
    def __init__(self, rows=5, cols=5, num_tribes=3, seed=None):
        self.rows = rows
        self.cols = cols
        self.num_tribes = num_tribes
        self.seed = seed

        # Initialize the core simulation environment
        self.sim = CivilizationSimulation_ENV(rows, cols, num_tribes)

        if seed is not None:
            random.seed(seed)

        self.num_agents = num_tribes                   # One agent per tribe
        self.action_space = [3] * self.num_agents      # Each agent has 3 possible actions
        self.observation_space = (rows, cols, 3)       # [population, food, tribe_id]
        self.last_scores = [0] * self.num_agents       # Used for computing reward deltas

    # =====================================================
    # Reset the environment and simulation to initial state
    # =====================================================
    def reset(self):
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes)
        self.last_scores = [0] * self.num_agents
        return self._get_obs()

    # =====================================================
    # Get structured observation of the current grid state
    # Returns: numpy array [rows x cols x 3] with [pop, food, tribe_id]
    # =====================================================
    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs

    # =====================================================
    # Execute the given actions, update the environment,
    # and return next state, reward, and done flag
    # =====================================================
    def step(self, actions):
        self.sim.take_turn(actions)          # Advance simulation
        obs = self._get_obs()                # Get new observation
        rewards = self._compute_rewards()    # Compute shaped rewards
        done = False                         # No termination logic yet
        return obs, rewards, done, {}

    # =====================================================
    # Compute rewards based on local and score-based shaping
    # local_reward = weighted sum of pop + food + survival
    # score_reward = delta in final score from previous step
    # mixed_reward = local_reward + λ × score_reward
    # =====================================================
    def _compute_rewards(self, lambda_score=0.1):
        local_rewards = [0] * self.num_agents
        prev_cells = self.last_cells if hasattr(self, 'last_cells') else [0] * self.num_agents

        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    idx = cell.tribe - 1
                    # Reward components: population, food, and presence bonus
                    local_rewards[idx] += 0.5 * cell.population + 0.2 * cell.food + 2.0

        # Score reward = improvement in final score since last step
        current_scores = self.compute_final_scores()
        score_rewards = [curr - prev for curr, prev in zip(current_scores, self.last_scores)]
        self.last_scores = current_scores

        # Combine both rewards
        mixed_rewards = [l + lambda_score * s for l, s in zip(local_rewards, score_rewards)]
        return mixed_rewards

    # =====================================================
    # Compute final performance score for each tribe
    # Combines: territory control, population, and food ratio
    # =====================================================
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

    # =====================================================
    # Print grid and simulation stats to console
    # Useful for debugging and visualization
    # =====================================================
    def render(self):
        self.sim.printGrid()
        self.sim.printDebugInfo()
        self.sim.printStats()
        print("Final scores:", self.compute_final_scores())

    # =====================================================
    # Return flattened version of the full grid state
    # Used by high-level manager in Hi-MAPPO
    # =====================================================
    def get_global_state(self):
        return self._get_obs().flatten()

    # =====================================================
    # Return duplicate global state for each agent
    # Used when agents share the same state as input
    # =====================================================
    def get_agent_obs(self):
        state = self._get_obs().flatten()
        return [state.copy() for _ in range(self.num_agents)]
