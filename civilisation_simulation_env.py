import random
import time
import os

# Define a cell on the civilization grid
class Cell:
    def __init__(self):
        self.population = 0     # number of people in the cell
        self.food = 0.0         # amount of food stored
        self.tribe = None       # tribe ID owning the cell

    def is_empty(self):
        # check if the cell is empty
        return self.population == 0

    def get_efficiency(self):
        # food generation efficiency (diminishing returns with more people)
        if self.population <= 0:
            return 0
        return sum(1 / (i + 1) for i in range(self.population))

# Civilization simulation environment
class CivilizationSimulation_ENV:
    def __init__(self, rows, cols, num_tribes, seed=None):
        if seed is not None:
            random.seed(seed)
        self.rows = rows
        self.cols = cols
        self.grid = [[Cell() for _ in range(cols)] for _ in range(rows)]
        self.num_tribes = num_tribes
        self.expansion_success = [0] * num_tribes
        self.population_before = [0] * num_tribes
        self.init_tribes()

    def init_tribes(self):
        # randomly place each tribe on the grid
        placed = 0
        while placed < self.num_tribes:
            i, j = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if self.grid[i][j].is_empty():
                self.grid[i][j].population = 2
                self.grid[i][j].food = 10.0
                self.grid[i][j].tribe = placed + 1
                placed += 1

    def neighbors(self, i, j):
        # yield coordinates of neighboring cells
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    yield ni, nj

    def take_turn(self, tribe_actions):
        """
        Execute one simulation step based on each tribe's action choice.

        Actions:
        - 0: harvest food based on efficiency
        - 1: grow population if enough food
        - 2: expand into neighboring empty cells
        Also handles starvation and updates the grid state.
        """
        self.expansion_success = [0] * self.num_tribes
        self.population_before = [0] * self.num_tribes
        self.expand_reward = [0.0] * self.num_tribes
        new_grid = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                c = self.grid[i][j]
                if c.tribe:
                    self.population_before[c.tribe - 1] += c.population

        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell.population > 0:
                    tribe_id = cell.tribe
                    action = tribe_actions[tribe_id - 1]

                    if action == 0:
                        cell.food += cell.get_efficiency()
                    elif action == 1:
                        if cell.food >= cell.population:
                            cell.food -= cell.population
                            cell.population += 1
                    elif action == 2:
                        if cell.food >= 5 and cell.population >= 2:
                            for ni, nj in self.neighbors(i, j):
                                if self.grid[ni][nj].is_empty():
                                    new_grid[ni][nj].population = 2
                                    new_grid[ni][nj].food = 5.0
                                    new_grid[ni][nj].tribe = cell.tribe
                                    cell.population -= 2
                                    cell.food -= 5
                                    self.expansion_success[tribe_id - 1] = 1
                                    break

                    # starvation logic
                    food_required = cell.population / 5
                    if cell.food < food_required:
                        deaths = min(cell.population, int((food_required - cell.food) * 5))
                        cell.population -= deaths
                        cell.food = 0
                    else:
                        cell.food -= food_required

                    if cell.population > 0 and new_grid[i][j].is_empty():
                        new_grid[i][j].population = cell.population
                        new_grid[i][j].food = cell.food
                        new_grid[i][j].tribe = cell.tribe

        self.grid = new_grid

    def printGrid(self):
        # print simple visualization of the grid
        print("\n=== Civilization Map ===")
        for row in self.grid:
            line = ""
            for cell in row:
                if cell.population > 0:
                    line += f"T{cell.tribe:<3}"
                else:
                    line += ".   "
            print(line)

    def printDebugInfo(self):
        # print detailed info of each cell
        print("\n=== Debug Info ===")
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell.population > 0:
                    print(
                        f"({i},{j}) Tribe {cell.tribe} | Pop: {cell.population} | "
                        f"Food: {round(cell.food, 2)} | Eff: {round(cell.get_efficiency(), 2)}"
                    )

    def printStats(self):
        # print summary stats for each tribe
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
