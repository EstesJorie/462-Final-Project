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
