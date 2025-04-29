import numpy as np
from civilisation_simulation_env import CivilizationSimulation_ENV

SEED = 7  # fixed random seed for reproducibility

# Civilization environment wrapper for QMIX agents
class CivilizationEnv_QMIX:
    def __init__(self, rows=5, cols=5, num_tribes=3):
        self.rows = rows
        self.cols = cols
        self.num_tribes = num_tribes

        self.sim = CivilizationSimulation_ENV(rows, cols, num_tribes, seed=SEED)
        self.num_agents = num_tribes
        self.action_space = [3] * self.num_agents
        self.observation_space = (rows, cols, 3)
        self.last_scores = [0] * self.num_agents

        # reward shaping weights
        self.weight_pop = 0.5
        self.weight_food = 0.05
        self.weight_survival = 2.0
        self.lambda_score = 0.3
        self.weight_expand = 1.0
        self.last_avg_score = 0.0

    def reset(self):
        # reset environment to a new simulation
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes)
        self.last_scores = [0] * self.num_agents
        return self._get_obs()

    def _get_obs(self):
        # return current grid observations
        obs = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs

    def step(self, actions):
        """
        Execute one environment step based on agent actions.

        - apply all tribes' actions simultaneously
        - update grid state: food, population, expansions
        - calculate reward signals for learning
        """
        self.sim.take_turn(actions)
        obs = self._get_obs()

        rewards, current_scores = self._compute_rewards()
        self._update_reward_weights(current_scores)

        done = False
        return obs, rewards, done, {}

    def _compute_rewards(self):
        """
        Calculate shaped rewards for each agent.

        Reward components:
        - local development: population, food collected, survival
        - expansion success: new territory gained
        - civilization score improvement
        """
        local_rewards = [0.0] * self.num_agents

        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    idx = cell.tribe - 1
                    food_term = min(cell.food, 10)  # cap food to avoid reward explosion
                    local_rewards[idx] += (
                        self.weight_pop * cell.population +
                        self.weight_food * (food_term ** 0.8) +
                        self.weight_survival
                    )

        expansion_bonus = self.sim.expand_reward
        current_scores = self.compute_final_scores()
        score_rewards = [curr - prev for curr, prev in zip(current_scores, self.last_scores)]
        self.last_scores = current_scores

        mixed_rewards = [
            local + self.weight_expand * expand + self.lambda_score * score
            for local, expand, score in zip(local_rewards, expansion_bonus, score_rewards)
        ]

        return mixed_rewards, current_scores

    def _update_reward_weights(self, current_scores):
        """
        Dynamically adjust reward shaping weights based on agent progress.

        - if average score improves: slightly increase reward weights
        - if stagnates or drops: slightly decay weights
        """
        avg_score = sum(current_scores) / len(current_scores)

        if avg_score > self.last_avg_score:
            self.weight_pop += 0.01
            self.weight_food += 0.05
            self.weight_survival += 0.05
            self.weight_expand += 0.02
            self.lambda_score += 0.005
        else:
            self.weight_pop *= 0.99
            self.weight_food *= 0.97
            self.weight_survival *= 0.98
            self.weight_expand *= 0.98
            self.lambda_score *= 0.99

        self.last_avg_score = avg_score

    def compute_final_scores(self, H=5, food_per_person=0.2):
        """
        Compute long-term civilization scores.

        Scores combine:
        - territory controlled (40%)
        - normalized population (40%)
        - food sufficiency over H turns (20%)

        Hoarding excess food is slightly penalized.
        """
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

    def render(self):
        # visualize the current grid state
        self.sim.printGrid()
        self.sim.printDebugInfo()
        self.sim.printStats()
        print("Final scores:", self.compute_final_scores())
