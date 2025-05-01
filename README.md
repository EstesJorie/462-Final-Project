  # Civilization Simulation with Multi-Agent Reinforcement Learning

This project simulates a grid-based civilization environment where multiple "tribes" act as agents to gather resources, grow populations, and expand territory. It supports training with three multi-agent reinforcement learning algorithms:

- MAPPO (Multi-Agent PPO)
- QMIX (Value Decomposition for Cooperative Learning)
- Hi-MAPPO (Hierarchical MAPPO with Monte Carlo Tree Search)
- Hi-MAPPO No MCTS (Hierarchical MAPPO with goal-driven planning)

Additionally, a random strategy baseline is included for performance comparison.

> **Note:** To contribute to this repository, please follow the steps below:
>
> - Fork the repository 
> - Clone the repository [https://github.com/[USERNAME]/[REPO_FORK_NAME].git](https://github.com/EstesJorie/462-Final-Project.git)
> - Create a new branch on cloned repository* 
>
>```bash 
>git checkout -b [new_branch]
>```
> - Make changes, updates, and/or fixes. Once complete open a merge request on the original repository with details of update/changes/fixes. 
>
>>*[YOUR_BRANCH ----PULL REQUEST---> ORIGINAL REPO (MAIN OR DEV)]*
>
> - Wait for your request to be approved!
>
> * Previously existing branch names such as *[Main or Dev]* should not be used!
---

## STEPS:

> In the terminal run *[(verbatim)]*
> ```bash
>conda env create -f environment.yml
>conda activate 462-env
>```

> 1. After cloning/downloading the repo, run *[train_all.py]*
> 2. Verify that the trained model folders exist, then run *[agentCompete.py]*
> 3. For analysis, run *[Analysis.py]*
> 4. For model evaluation, run *[evaluate_all_model.py]*

## Action Space

Each tribe (agent) can choose one of the following actions per step:

- `0`: Harvest – Gather food based on efficiency  
- `1`: Grow – Increase population if enough food is available  
- `2`: Expand – Spread to neighboring cells if population and food requirements are met  

Each episode consists of 10 simulation steps. The goal is to maximize:

- Total population  
- Food reserves  
- Number of occupied grid cells  

---

## Supported Algorithms

| Algorithm    | Description |
|--------------|-------------|
| **MAPPO**    | Each agent has an independent actor network and shares a centralized critic; trained via PPO |
| **QMIX**     | Centralized training with decentralized execution using value decomposition |
| **Hi-MAPPO** | A hierarchical framework where a high-level policy (Hi-MAPPO) sets strategic goals, and low-level planning uses **Monte Carlo Tree Search (MCTS)** to explore and select optimal action sequences to achieve those goals |
| **Hi-MAPPO without MCTS** | A hierarchical approach where a high-level manager selects goals (harvest/grow/expand) and low-level workers execute actions |

Each algorithm supports full training loops, model saving, and reloading.

--- 
## Contributors

[EstesJorie](https://github.com/EstesJorie)
[Kapibaris](https://github.com/Kapibaris)
[zleihupo](https://github.com/zleihupo)

---
