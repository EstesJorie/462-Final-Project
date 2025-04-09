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
        # Diminishing returns per person (to promote expansion)
        if self.population <= 0:
            return 0
        return sum(1 / (i + 1) for i in range(self.population))

class CivilizationSimulation:
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
            if self.grid[i][j].is_empty():    #arbitrary starting values
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

    def take_turn(self):
        new_grid = [[Cell() for _ in range(self.cols)] for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(self.cols):
                cell = self.grid[i][j]
                if cell.population > 0:
                    actions = []
                    if cell.food >= cell.population:  # Enough food to feed
                        actions.append("grow")
                    if cell.food >= 5 and cell.population >= 4:
                        for ni, nj in self.neighbors(i, j):
                            if self.grid[ni][nj].is_empty():
                                actions.append("expand")
                                break
                    actions.append("gather")  # Always can gather

                    action = random.choice(actions)

                    # Perform chosen action
                    if action == "gather":   #gather food (diminishing returns)
                        food_gained = cell.get_efficiency()
                        cell.food += food_gained

                    elif action == "grow":   #inc. population for arbitrary food value
                        if cell.food >= cell.population:
                            cell.food -= cell.population
                            cell.population += 1

                    elif action == "expand":  #expand to new cell- arbitrary conditions
                        for ni, nj in self.neighbors(i, j):
                            neighbor = self.grid[ni][nj]
                            if neighbor.is_empty():
                                new_grid[ni][nj].population = 2
                                new_grid[ni][nj].food = 5.0
                                new_grid[ni][nj].tribe = cell.tribe
                                cell.population -= 2
                                cell.food -= 5
                                break

                    # Feed population
                    food_required = cell.population/5   #arbitrary- reduced to make sure evolves
                    if cell.food < food_required:
                        starvation = food_required - cell.food
                        deaths = min(cell.population, int(starvation*5))
                        cell.population -= deaths
                        cell.food = 0
                    else:
                        cell.food -= food_required

                    if cell.population <= 0:
                        continue  # Cell dies off

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

    def runSimulation(self, generations):
        for gen in range(generations):
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"\nGeneration {gen + 1}")
            self.take_turn()
            self.printGrid()
            self.printDebugInfo()
            self.printStats()
            time.sleep(0.1)

class GameController:
    @staticmethod
    def getValidDimensions():
        while True:
            try:
                rows, cols = map(int, input("Enter no. of rows and columns: ").split())
                if rows >= 3 and cols >= 3:
                    return rows, cols
                else:
                    print(f"Invalid input please enter values greater than or equal to 3.")
            except ValueError:
                print("Please enter two integers separated by space.")

    @staticmethod
    def getValidGenerations():
        while True:
            try:
                generations = int(input("Enter number of generations: "))
                if generations > 1:
                    return generations
                else:
                    print(f"Number of generations equals {generations}. Enter a value greater than 1 to continue.")
            except ValueError:
                print("Please enter a valid integer.")

    @staticmethod
    def getValidTribeCount():
        while True:
            try:
                tribes = int(input("Enter number of starting tribes: "))
                if tribes >= 1:
                    return tribes
                else:
                    print("Please enter a value of 1 or more.")
            except ValueError:
                print("Please enter a valid integer.")

def main():
    controller = GameController()
    rows, cols = controller.getValidDimensions()
    generations = controller.getValidGenerations()
    num_tribes = controller.getValidTribeCount()

    game = CivilizationSimulation(rows, cols, num_tribes)
    game.runSimulation(generations)

if __name__ == "__main__":
    main()
