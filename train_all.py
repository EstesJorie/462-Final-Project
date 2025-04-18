from train_hi_mappo import train_hi_mappo
from train_qmix import train_qmix
from train_mappo import train_mappo
from GameController import GameController
import time

TEST_CONFIG = {
        'rows': 5,
        'cols': 5,
        'generations': 1000,
        'num_tribes': 3
    }

def getModeSelection():
    """
    Returns:
        bool: True if TEST_MODE, False if USER_MODE
    """ 
    while True:
          print(f"Select mode:\n")
          print("1. TEST MODE (Uses preset values)\n")
          print("2. USER MODE (Input values)\n")
          print("q. Shows TEST_CONFIG\n")
          print("Enter '1' for TEST MODE, '2' for USER MODE, or 'q' to see TEST CONFIG.\n")
          choice = input("Enter choice: ").strip().lower()
          if choice in ['1', '2']:
               return choice == '1' # if choice = 1, then TRUE thus TEST_MODE = True
          elif choice == 'q':
               print(f"{TEST_CONFIG}\n")
               print("Do you want to update the TEST_CONFIG values? (y/n)\n")
               updatedChoice = input("Enter choice: ").strip().lower()
               if updatedChoice == 'y':
                    print("Updating TEST_CONFIG values...\n")
                    rows = int(input("Enter number of rows: "))
                    cols = int(input("Enter number of columns: "))
                    generations = int(input("Enter number of generations: "))
                    num_tribes = int(input("Enter number of tribes: "))
                    TEST_CONFIG['rows'] = rows
                    TEST_CONFIG['cols'] = cols
                    TEST_CONFIG['generations'] = generations
                    TEST_CONFIG['num_tribes'] = num_tribes
               continue
          raise ValueError("Invalid choice. Please enter 1, 2, or q.\n")

def trainAllModels(rows=None, cols=None, generations=None, num_tribes=None):
    """
    Args: (optional)
        rows (int) - Num of grid rows
        cols (int) - Num of grid cols
        generations (int) - Num of generations to train
        num_tribes (int) - Num of tribes to train
        test_mode (bool) - If True, use preset values for testing
    Returns:
        tuple(mappo, hi_mappo, qmix) - Trained agents

    Raises:
        Exception: If training fails for any model
    """
    TEST_MODE = getModeSelection()

    if TEST_MODE:
        rows = TEST_CONFIG['rows']
        cols = TEST_CONFIG['cols']
        generations = TEST_CONFIG['generations']
        num_tribes = TEST_CONFIG['num_tribes']
        print(f"\n===TEST MODE===\n")
        print(f"> Rows: {rows}\n > Cols: {cols}\n > Generations: {generations}\n > Tribes: {num_tribes}\n")
    else:
    # default value setting
        if any(param is None for param in [rows, cols, generations, num_tribes]):
            controller = GameController() 
            rows = rows or controller.getValidDimensions()[0] #if rows is not None keep, else use val 0
            cols = cols or controller.getValidDimensions()[1] #if cols is not None keep, else use val 1
            generations = controller.getValidGenerations()
            num_tribes = controller.getValidTribeCount()
    try:
        print("\n=== Training MAPPO ===\n")
        mappo = train_mappo(
            rows=rows,
            cols=cols,
            generations=generations,
            num_tribes=num_tribes
        )
        print("MAPPO training completed.\n")
        time.sleep(1)

        print("\n=== Training Hi-MAPPO ===\n")
        hi_mappo = train_hi_mappo(
            rows=rows,
            cols=cols,
            generations=generations,
            num_tribes=num_tribes
        )
        print("Hi-MAPPO training completed.\n")
        time.sleep(1))

        print("\n=== Training QMIX ===\n")
        qmix = train_qmix(
            rows=rows,
            cols=cols,
            generations=generations,
            num_tribes=num_tribes
        )
        print("QMIX training completed.\n")
        time.sleep(1)

        print("\n=== All models trained successfully! ===\n")
        return mappo, hi_mappo, qmix
    except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    trainAllModels()

