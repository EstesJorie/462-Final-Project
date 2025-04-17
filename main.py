from CivilisationSimulation2 import all
from GameController import GameController

def main():
    controller = GameController()
    rows, cols = controller.getValidDimensions()
    generations = controller.getValidGenerations()
    num_tribes = controller.getValidTribeCount()

    game = CivilisationSimulation(rows, cols, num_tribes)
    game.runSimulation(generations)

if __name__ == "__main__":
    main()