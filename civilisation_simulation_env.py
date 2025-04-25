import random
import time
import os

# Define a cell on the civilization grid
class Cell:
    def __init__(self):
        self.population = 0     # Number of people in the cell
        self.food = 0.0         # Amount of food stored in the cell
        self.tribe = None       # ID of the tribe that owns this cell

    # Check if the cell is empty (no population)
    def is_empty(self):
        return self.population == 0

    # Calculate the cell's productivity (food generation efficiency)
    def get_efficiency(self):
        if self.population <= 0:
            return 0
        # Returns the harmonic sum of population (diminishing returns)
        return sum(1 / (i + 1) for i in range(self.population))

# Civilization simulation environment with ENV interface
class CivilizationSimulation_ENV:
    def __init__(self, rows, cols, num_tribes, seed=None):
        # Initialize random seed for reproducibility
        if seed is not None:
            random.seed(seed)
        self.rows = rows                          # Number of rows in the grid
        self.cols = cols                          # Number of columns in the grid
        self.grid = [[Cell() for _ in range(cols)] for _ in range(rows)]  # 2D grid of cells
        self.num_tribes = num_tribes              # Number of tribes in the simulation
        self.expansion_success = [0] * num_tribes # Track expansion success for each tribe
        self.population_before = [0] * num_tribes # Track previous population
        self.init_tribes()                        # Randomly initialize tribes on the grid

    # Place each tribe randomly on the grid
    def init_tribes(self):
        placed = 0
        while placed < self.num_tribes:
            i, j = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if self.grid[i][j].is_empty():
                self.grid[i][j].population = 2    # Initial population
                self.grid[i][j].food = 10.0       # Initial food supply
                self.grid[i][j].tribe = placed + 1
                placed += 1

    # Generator that yields valid neighbor coordinates for cell (i, j)
    def neighbors(self, i, j):
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # Skip the current cell
                ni, nj = i + di, j + dj
                if 0 <= ni < self.rows and 0 <= nj < self.cols:
                    yield ni, nj

    # Perform one simulation step based on the tribe actions
    def take_turn(self, tribe_actions):
        self.expansion_success = [0] * self.num_tribes
        self.population_before = [0] * self.num_tribes
        self.expand_reward = [0.0] * self.num_tribes  # Reward given for successful or attempted expansions
        new_grid = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]

        # Record population before actions
        for i in range(self.rows):
            for j in range(self.cols):
                c = self.grid[i][j]
                if c.tribe:
                    self.population_before[c.tribe - 1] += c.population

        # Execute each tribe's action
        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell.population > 0:
                    tribe_id = cell.tribe
                    action = tribe_actions[tribe_id - 1]  # Get action for this tribe

                    # Action 0: Harvest food based on efficiency
                    if action == 0:
                        cell.food += cell.get_efficiency()

                    # Action 1: Grow population if enough food is available
                    elif action == 1:
                        if cell.food >= cell.population:
                            cell.food -= cell.population
                            cell.population += 1

                    # Action 2: Attempt to expand into neighboring cells
                    elif action == 2:
                        if cell.food >= 5 and cell.population >= 2:  # Relaxed expansion condition
                            for ni, nj in self.neighbors(i, j):
                                if self.grid[ni][nj].is_empty():
                                    # Place a new cell for the tribe
                                    new_grid[ni][nj].population = 2
                                    new_grid[ni][nj].food = 5.0
                                    new_grid[ni][nj].tribe = cell.tribe
                                    cell.population -= 2
                                    cell.food -= 5
                                    self.expansion_success[tribe_id - 1] = 1
                                    break

                    # Handle starvation logic
                    food_required = cell.population / 5
                    if cell.food < food_required:
                        # Not enough food: lose population
                        deaths = min(cell.population, int((food_required - cell.food) * 5))
                        cell.population -= deaths
                        cell.food = 0
                    else:
                        cell.food -= food_required

                    # Transfer cell to new grid if still has population
                    if cell.population > 0 and new_grid[i][j].is_empty():
                        new_grid[i][j].population = cell.population
                        new_grid[i][j].food = cell.food
                        new_grid[i][j].tribe = cell.tribe

        # Update grid with new state
        self.grid = new_grid

    # Print a simplified visual representation of the grid
    def printGrid(self):
        print("\n=== Civilization Map ===")
        for row in self.grid:
            line = ""
            for cell in row:
                if cell.population > 0:
                    symbol = f"T{cell.tribe}"      # Mark cell with tribe ID
                    line += f"{symbol:<4}"
                else:
                    line += ".   "                 # Empty cell
            print(line)

    # Print internal debug info for each cell with population
    def printDebugInfo(self):
        print("\n=== Debug Info ===")
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell.population > 0:
                    print(
                        f"({i},{j}) Tribe {cell.tribe} | Pop: {cell.population} | "
                        f"Food: {round(cell.food, 2)} | Eff: {round(cell.get_efficiency(), 2)}"
                    )

    # Print summary statistics for each tribe
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

        # Print sorted stats for each tribe
        for tribe, data in sorted(tribe_data.items()):
            print(f"Tribe {tribe}: Population = {data['pop']} | Food = {round(data['food'], 2)}")
