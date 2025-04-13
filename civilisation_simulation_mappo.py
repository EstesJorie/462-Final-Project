import random
import time
import os

class Cell:
    def __init__(self):
        self.population = 0
        self.food = 0.0
        self.tribe = None

    def is_empty(self):
        return self.population == 0

    def get_efficiency(self):
        if self.population <= 0:
            return 0
        return sum(1 / (i + 1) for i in range(self.population))

class CivilizationSimulation_MAPPO:
    def __init__(self, rows, cols, num_tribes):
        self.rows = rows
        self.cols = cols
        self.grid = [[Cell() for _ in range(cols)] for _ in range(rows)]
        self.num_tribes = num_tribes
        self.init_tribes()

    def init_tribes(self):
        placed = 0
        while placed < self.num_tribes:
            i, j = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if self.grid[i][j].is_empty():
                self.grid[i][j].population = 2
                self.grid[i][j].food = 10.0
                self.grid[i][j].tribe = placed + 1
                placed += 1

    def neighbors(self, i, j):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    yield ni, nj

    def take_turn(self, tribe_actions):
        new_grid = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell.population > 0:
                    tribe_id = cell.tribe
                    action = tribe_actions[tribe_id - 1]
                    if action == 0:  # gather
                        food_gained = cell.get_efficiency()
                        cell.food += food_gained
                    elif action == 1:  # grow
                        if cell.food >= cell.population:
                            cell.food -= cell.population
                            cell.population += 1
                    elif action == 2:  # expand
                        if cell.food >= 5 and cell.population >= 4:
                            for ni, nj in self.neighbors(i, j):
                                if self.grid[ni][nj].is_empty():
                                    new_grid[ni][nj].population = 2
                                    new_grid[ni][nj].food = 5.0
                                    new_grid[ni][nj].tribe = cell.tribe
                                    cell.population -= 2
                                    cell.food -= 5
                                    break

                    food_required = cell.population / 5
                    if cell.food < food_required:
                        starvation = food_required - cell.food
                        deaths = min(cell.population, int(starvation * 5))
                        cell.population -= deaths
                        cell.food = 0
                    else:
                        cell.food -= food_required
                    if cell.population <= 0:
                        continue
                    if new_grid[i][j].is_empty():
                        new_grid[i][j].population = cell.population
                        new_grid[i][j].food = cell.food
                        new_grid[i][j].tribe = cell.tribe
        self.grid = new_grid

    def printGrid(self):
        print("\n=== Civilization Map ===")
        for row in self.grid:
            line = ""
            for cell in row:
                if cell.population > 0:
                    symbol = f"T{cell.tribe}"
                    line += f"{symbol:<4}"
                else:
                    line += ".   "
            print(line)

    def printDebugInfo(self):
        print("\n=== Debug Info ===")
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell.population > 0:
                    print(
                        f"({i},{j}) Tribe {cell.tribe} | Pop: {cell.population} | "
                        f"Food: {round(cell.food, 2)} | Eff: {round(cell.get_efficiency(), 2)}"
                    )

    def printStats(self):
        print("\n=== Tribe Stats Summary ===")
        tribe_data = {}
        for row in self.grid:
            for cell in row:
                if cell.population > 0:
                    if cell.tribe not in tribe_data:
                        tribe_data[cell.tribe] = {"pop": 0, "food": 0}
                    tribe_data[cell.tribe]["pop"] += cell.population
                    tribe_data[cell.tribe]["food"] += cell.food

        for tribe, data in sorted(tribe_data.items()):
            print(f"Tribe {tribe}: Population = {data['pop']} | Food = {round(data['food'], 2)}")
