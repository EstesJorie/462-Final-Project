from GameOfLife import GameOfLife
from GameController import GameController

def main():
    controller = GameController()
    rows, cols = controller.getValidDimensions()
    generations = controller.getValidGenerations()
    
    game = GameOfLife(rows, cols)
    game.runSimulation(generations)


if __name__ == "__main__":
    main()
