import random
import time
import os

def findNextGen(mat):
    m = len(mat)
    n = len(mat[0])
    nextGen = [[0 for _ in range(n)] for _ in range(m)]

    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for i in range(m):
        for j in range(n):
            liveNeighbours = 0

            for dir in directions: #count live neighhbours
                ni = i + dir[0]
                nj = j + dir[1]
                if 0 <= ni < m and 0 <= nj < n and mat[ni][nj]:
                    liveNeighbours += 1

            if mat[i][j] == 1 and (liveNeighbours == 2 or liveNeighbours == 3):
                nextGen[i][j] = 1 #cell lives
            elif mat[i][j] == 0 and liveNeighbours == 3:
                nextGen[i][j] = 1 #cell becomes live
            else:
                nextGen[i][j] = 0 #cell dies
    
    return nextGen

def initRandomGrid(m, n): #random grid 
    return [[random.randint(0, 1) for _ in range(n)] for _ in range(m)]

def printGrid(mat): #print current grid
    for row in mat:
        print(" ".join('*' if cell == 1 else '.' for cell in row ))
    print()

def main():
    m, n = map(int, input("Enter no. of rows and columns: ").split())

    if m < 3 or n < 3: #input validation
        print(f"Invalid input please enter a value greater than {m},{n}. ")
        m, n = map(int, input("Enter no. of rows and columns: ").split())
    
    mat = initRandomGrid(m, n)
    gens = int(input("Enter number of generations: "))

    if gens <= 1: #generation validation
        print(f"Number of generations equals {gens}. Enter a value greater to continue.")
        gens = int(input("Enter number of generations: "))

    for gen in range(gens):
        os.system('cls' if os.name == 'nt' else 'clear') #clear console
        print(f"Generation {gen + 1}:")
        printGrid(mat) #print current grid
        mat = findNextGen(mat) #find next generation 
        time.sleep(0.75) #.75 sec pause between generations

if __name__ == "__main__":
    main()

