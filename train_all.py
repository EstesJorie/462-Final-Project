from train_hi_mappo import train_hi_mappo
from train_qmix import train_qmix
from train_mappo import train_mappo
from GameController import GameController
import time

def trainAllModels(rows=None, cols=None, generations=None, num_tribes=None):
    """
    Args: (optional)
        rows (int) - Num of grid rows
        cols (int) - Num of grid cols
        generations (int) - Num of generations to train
        num_tribes (int) - Num of tribes to train

    Returns:
        tuple(mappo, hi_mappo, qmix) - Trained agents

    Raises:
        Exception: If training fails for any model
    """

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
        time.sleep(2)

        print("\n=== Training Hi-MAPPO ===\n")
        hi_mappo = train_hi_mappo(
            rows=rows,
            cols=cols,
            generations=generations,
            num_tribes=num_tribes
        )
        print("Hi-MAPPO training completed.\n")
        time.sleep(2)

        print("\n=== Training QMIX ===\n")
        qmix = train_qmix(
            rows=rows,
            cols=cols,
            generations=generations,
            num_tribes=num_tribes
        )
        print("QMIX training completed.\n")
        time.sleep(2)

        print("\n=== All models trained successfully ===\n")
        return mappo, hi_mappo, qmix
    except Exception as e:
            print(f"Training failed: {str(e)}")
            raise

if __name__ == "__main__":
    trainAllModels(rows=5, cols=5, generations=1000, num_tribes=3)

