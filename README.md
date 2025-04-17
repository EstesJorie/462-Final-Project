
# 🧠 Civilization Simulation with Multi-Agent Reinforcement Learning

This project simulates a grid-based civilization environment where multiple "tribes" act as agents to gather resources, grow populations, and expand territory. It supports training with three multi-agent reinforcement learning algorithms:

- MAPPO (Multi-Agent PPO)
- QMIX (Value Decomposition for Cooperative Learning)
- Hi-MAPPO (Hierarchical MAPPO with goal-driven planning)

Additionally, a random strategy baseline is included for performance comparison.

---

## 📁 Project Structure

```
.
├── civilisation_simulation_env.py         # Core simulation environment (grid, tribes, actions)
├── civilization_env_mappo.py             # MAPPO-compatible environment wrapper
├── civilization_env_qmix.py              # QMIX-compatible environment wrapper
├── civilization_env_hi_mappo.py          # Hi-MAPPO-compatible environment wrapper
├── mappo.py                              # MAPPO agent implementation
├── qmix.py                               # QMIX agent implementation
├── hi_mappo.py                           # Hi-MAPPO agent (manager + workers)
├── train_mappo.py                        # MAPPO training script
├── train_qmix.py                         # QMIX training script
├── train_hi_mappo.py                     # Hi-MAPPO training script
├── evaluate_all_model.py                 # Evaluation script comparing MAPPO/QMIX/Hi-MAPPO/Random
```

---

## 🕹️ Action Space

Each tribe (agent) can choose one of the following actions per step:

- `0`: Harvest – Gather food based on efficiency  
- `1`: Grow – Increase population if enough food is available  
- `2`: Expand – Spread to neighboring cells if population and food requirements are met  

Each episode consists of 10 simulation steps. The goal is to maximize:

- Total population  
- Food reserves  
- Number of occupied grid cells  

---

## 🤖 Supported Algorithms

| Algorithm    | Description |
|--------------|-------------|
| **MAPPO**    | Each agent has an independent actor network and shares a centralized critic; trained via PPO |
| **QMIX**     | Centralized training with decentralized execution using value decomposition |
| **Hi-MAPPO** | A hierarchical approach where a high-level manager selects goals (harvest/grow/expand) and low-level workers execute actions |

Each algorithm supports full training loops, model saving, and reloading.

---

## 📈 Evaluation

Run the following command to compare all agents' performance:

```bash
python evaluate_all_model.py
```

You will be prompted to enter:
- Grid size (e.g. `5 5`)
- Number of generations (e.g. `1000`)
- Number of initial tribes (e.g. `3`)

The script generates plots comparing:
- Total population and food across methods
- Occupied cell count (territory expansion)
- Output image: `evaluate_all_model_smoothed.png`

---

## 🏁 Training

Run one of the following scripts to train the respective agent:

```bash
python train_all.py
```

The script can be configured to 

Model save directories:
- `trained_models/` (MAPPO)
- `trained_models_qmix/` (QMIX)
- `trained_models_hi_mappo/` (Hi-MAPPO)

---

## 💡 Customization Tips

- Modify reward functions in the `_compute_rewards()` method of each `*_env_*.py`
- Add new actions or goals by extending `hi_mappo.py` and the environment
- Tweak network structures and training parameters in `mappo.py`, `qmix.py`, `hi_mappo.py`
- The random strategy may perform relatively well by opportunistically expanding into available cells.
- To improve the performance of learned policies, consider reward shaping or rule-based enhancements to better guide agents toward intended behaviors.

---
