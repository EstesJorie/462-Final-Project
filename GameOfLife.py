import random
import time
import os
import readchar

class GameOfLife:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = self.initRandomGrid()
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        self.generation = 0
    
    def initRandomGrid(self):
        return [[random.randint(0, 1) for _ in range(self.cols)] for _ in range(self.rows)]
    
    def printGrid(self):
        for row in self.grid:
            print(" ".join('*' if cell == 1 else '.' for cell in row))
        print()
    
    def countLiveNeighbors(self, i, j):
        live_neighbors = 0
        for di, dj in self.directions: #loop through neighbour directions
            ni, nj = i + di, j + dj
            if 0 <= ni < self.rows and 0 <= nj < self.cols and self.grid[ni][nj]: #valid neighbour
                live_neighbors += 1 #increment live neighbours
        return live_neighbors
    
    def calculateNextGeneration(self):
        next_gen = [[0 for _ in range(self.cols)] for _ in range(self.rows)] #create single row, repeat for number of rows
        
        for i in range(self.rows):
            for j in range(self.cols):
                live_neighbors = self.countLiveNeighbors(i, j)
                
                if self.grid[i][j] == 1 and (live_neighbors == 2 or live_neighbors == 3):
                    next_gen[i][j] = 1  # Cell survives
                elif self.grid[i][j] == 0 and live_neighbors == 3:
                    next_gen[i][j] = 1  # Cell becomes alive
                else:
                    next_gen[i][j] = 0  # Cell dies
        
        self.grid = next_gen
        self.generation += 1
    
    def runSimulation(self, generations):
        for gen in range(generations):
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear console
            print(f"Generation {gen + 1}:")
            
            if gen % 9 == 0 and gen > 0:  # Prompt to continue every 10 generations
                print("Press 'Q' to quit or any other key to continue.")
                k = readchar.readchar()
                if k.lower() == 'q':
                    break
            
            self.printGrid()
            self.calculateNextGeneration()
            time.sleep(0.75)  # 0.75 second pause between generations