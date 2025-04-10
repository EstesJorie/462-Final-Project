# -*- coding: utf-8 -*-
"""Civilisation_Sim_Env.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qbeujI2rzr6kQfA6LRE6TeVGOrPp2sId
"""

import numpy as np
import random
from civilisation_simulation_2 import CivilizationSimulation

ACTIONS = ["gather", "grow", "expand"]

class CivilizationEnv:
    def __init__(self, rows=5, cols=5, num_tribes=2, max_steps=100):
        self.rows = rows
        self.cols = cols
        self.num_tribes = num_tribes
        self.max_steps = max_steps
        self.sim = CivilizationSimulation(rows, cols, num_tribes)
        self.agents = [f"tribe_{i+1}" for i in range(num_tribes)]
        self.current_step = 0

    def reset(self):
        self.sim = CivilizationSimulation(self.rows, self.cols, self.num_tribes)
        self.current_step = 0
        return self._get_obs()

    def step(self, actions):
        self.current_step += 1

        # Apply actions for each agent
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.sim.grid[i][j]
                if cell.population > 0 and cell.tribe is not None:
                    agent_id = f"tribe_{cell.tribe}"
                    action = actions.get(agent_id, random.choice(ACTIONS))

                    if action == "gather":
                        food_gained = cell.get_efficiency()
                        cell.food += food_gained

                    elif action == "grow":
                        if cell.food >= cell.population:
                            cell.food -= cell.population
                            cell.population += 1

                    elif action == "expand":
                        for ni, nj in self.sim.neighbors(i, j):
                            neighbor = self.sim.grid[ni][nj]
                            if neighbor.is_empty() and cell.food >= 5 and cell.population >= 2:
                                neighbor.population = 2
                                neighbor.food = 5.0
                                neighbor.tribe = cell.tribe
                                cell.population -= 2
                                cell.food -= 5
                                break

                    # Feed population
                    food_required = cell.population / 5
                    if cell.food < food_required:
                        deaths = min(cell.population, int((food_required - cell.food) * 5))
                        cell.population -= deaths
                        cell.food = 0
                    else:
                        cell.food -= food_required

                    # Cleanup
                    if cell.population <= 0:
                        cell.population = 0
                        cell.food = 0
                        cell.tribe = None

        obs = self._get_obs()
        rewards = self._get_rewards()
        done = self._get_done()
        infos = {agent: {} for agent in self.agents}

        return obs, rewards, done, infos

    def _get_obs(self):
        obs = {}
        for agent in self.agents:
            tribe_id = int(agent.split("_")[1])
            pop, food, empty_neighbors = 0, 0, 0
            for i in range(self.rows):
                for j in range(self.cols):
                    cell = self.sim.grid[i][j]
                    if cell.tribe == tribe_id:
                        pop += cell.population
                        food += cell.food
                        empty_neighbors += sum(
                            1 for ni, nj in self.sim.neighbors(i, j)
                            if self.sim.grid[ni][nj].is_empty()
                        )
            obs[agent] = np.array([pop, food, empty_neighbors], dtype=np.float32)
        return obs

    def _get_rewards(self):
        rewards = {}
        for agent in self.agents:
            tribe_id = int(agent.split("_")[1])
            pop, food = 0, 0
            for i in range(self.rows):
                for j in range(self.cols):
                    cell = self.sim.grid[i][j]
                    if cell.tribe == tribe_id:
                        pop += cell.population
                        food += cell.food
            rewards[agent] = pop + 0.1 * food  # 可调整
        return rewards

    def _get_done(self):
        done_flag = self.current_step >= self.max_steps
        return {agent: done_flag for agent in self.agents}

    def render(self):
        self.sim.printGrid()
        self.sim.printStats()