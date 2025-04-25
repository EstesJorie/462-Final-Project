import random
import numpy as np
from civilisation_simulation_env import CivilizationSimulation_ENV

SEED = 7  # Fixed seed for reproducibility

class CivilizationEnv_HiMAPPO:
    def __init__(self, rows=5, cols=5, num_tribes=3, seed=None):
        self.rows = rows
        self.cols = cols
        self.num_tribes = num_tribes
        self.seed = seed

        self.sim = CivilizationSimulation_ENV(rows, cols, num_tribes)

        if seed is not None:
            random.seed(seed)

        self.num_agents = num_tribes
        self.action_space = [3] * self.num_agents
        self.observation_space = (rows, cols, 3)
        self.last_scores = [0] * self.num_agents

        self.weight_pop = 1.5
        self.weight_food = 0.05
        self.weight_survival = 2.0
        self.lambda_score = 0.3
        self.weight_expand = 2.0
        self.last_avg_score = 0.0

    def reset(self):
        self.sim = CivilizationSimulation_ENV(self.rows, self.cols, self.num_tribes)
        self.last_scores = [0] * self.num_agents
        return self._get_obs()

    def _get_obs(self):
        obs = np.zeros((self.rows, self.cols, 3))
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.sim.grid[i][j]
                obs[i][j] = [c.population, c.food, c.tribe if c.tribe else 0]
        return obs

    def step(self, actions, goals=None):
        self.sim.take_turn(actions)
        obs = self._get_obs()
        rewards = self._compute_rewards(goals)
        done = False
        return obs, rewards, done, {}

    def _compute_rewards(self, goals=None):
        local_rewards = [0.0] * self.num_agents
        population_now = [0] * self.num_agents

        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.tribe:
                    tid = cell.tribe - 1
                    local_rewards[tid] += (
                        self.weight_pop * cell.population +
                        self.weight_food * (cell.food ** 0.8) +
                        self.weight_survival
                    )
                    population_now[tid] += cell.population

        pop_growth = [max(0, now - before) for now, before in zip(population_now, self.sim.population_before)]
        growth_bonus = [0.0] * self.num_agents
        expand_bonus = [0.0] * self.num_agents

        if goals is not None:
            for i in range(self.num_agents):
                if goals[i] == 1 and pop_growth[i] > 0:
                    growth_bonus[i] = 5.0  # ✅ Manager 选 grow 且成功增长
                if goals[i] == 2 and self.sim.expansion_success[i] == 1:
                    expand_bonus[i] = 10.0  # ✅ Manager 选 expand 且扩张成功

        expansion_bonus = [e * 10.0 for e in self.sim.expand_reward]
        current_scores = self.compute_final_scores()
        score_rewards = [curr - prev for curr, prev in zip(current_scores, self.last_scores)]
        self.last_scores = current_scores

        mixed_rewards = [
            local + self.weight_expand * expand + self.lambda_score * score + grow + expand_succeed
            for local, expand, score, grow, expand_succeed in zip(
                local_rewards, expansion_bonus, score_rewards, growth_bonus, expand_bonus)
        ]

        return mixed_rewards

    def _update_reward_weights(self, current_scores):
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
        self.sim.printGrid()
        self.sim.printDebugInfo()
        self.sim.printStats()
        print("Final scores:", self.compute_final_scores())

    def get_global_state(self):
        return self._get_obs().flatten()

    def get_agent_obs(self):
        state = self._get_obs().flatten()
        return [state.copy() for _ in range(self.num_agents)]