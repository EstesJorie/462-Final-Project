from mappo import MAPPOAgent
from hi_mappo import HiMAPPOAgent
from qmix import QMIXAgent
from mixedAgentSimulator import MixedAgentSimulator, RandomAgent
from civilisation_simulation_env_mixed import CivilisationSimulationMixed
import logging
import os
import gc
import torch
import pandas as pd
import time

rows, cols = 10, 10
obs_dim = rows * cols * 3
act_dim = 3

fileName = "agentCompete.py"
outputFile = "mixed_agent_scores.csv"
if not os.path.exists(fileName):
    print(f"File '{fileName}' does not exist, creating a new one.\n")
    
logFile = "agentCompete.log"
if not os.path.exists(logFile):
    print(f"Log file '{logFile}' does not exist, creating a new one.\n")

logging.basicConfig(filename=logFile,
                filemode='a',
                format='%(asctime)s - %(levelname)s - %(message)s',
                level=logging.INFO,
                datefmt='%Y-%m-%d %H:%M:%S')
print(f"Logging initialized for {fileName} . Check {logFile} for details.\n")

# === Load MAPPO Agent ===
mappo = MAPPOAgent(obs_dim=obs_dim, act_dim=act_dim, num_agents=1)
mappo.actors[0].load_state_dict(torch.load("trained_models_mappo/actor_0.pth"))
mappo.critic.load_state_dict(torch.load("trained_models_mappo/critic.pth"))
logging.info(f"MAPPO agent loaded from trained_models_mappo/actor_0.pth and trained_models_mappo/critic.pth")
print(f"MAPPO agent loaded from trained_models_mappo/actor_0.pth and trained_models_mappo/critic.pth\n")

# === Load Hi-MAPPO Agent ===
himappo = HiMAPPOAgent(obs_dim=obs_dim, act_dim=3, num_agents=1, state_dim=obs_dim, goal_dim=3)
himappo.workers[0].load_state_dict(torch.load("trained_models_hi_mappo/worker_0.pth"))
himappo.manager.load_state_dict(torch.load("trained_models_hi_mappo/manager.pth"))
logging.info(f"Hi-MAPPO agent loaded from trained_models_hi_mappo/worker_0.pth and trained_models_hi_mappo/manager.pth")
print(f"Hi-MAPPO agent loaded from trained_models_hi_mappo/worker_0.pth and trained_models_hi_mappo/manager.pth\n")

# === Load QMIX Agent ===
qmix = QMIXAgent(
    obs_dim=300,
    state_dim=300,
    act_dim=3,
    n_agents=5,          
    hidden_dim=64,       
    buffer_size=10000,
    batch_size=64,
    lr=1e-3,
    gamma=0.99
)
qmix.agent_nets[0].load_state_dict(torch.load("trained_models_qmix/qmix_agent_0.pth"))
qmix.mix_net.load_state_dict(torch.load("trained_models_qmix/qmix_mixer.pth"))
logging.info(f"QMIX agent loaded from trained_models_qmix/qmix_agent_0.pth and trained_models_qmix/qmix_mixer.pth")
print(f"QMIX agent loaded from trained_models_qmix/qmix_agent_0.pth and trained_models_qmix/qmix_mixer.pth\n")

random = RandomAgent(obs_dim=obs_dim, act_dim=act_dim, rows=rows, cols=cols, num_tribes=3)
logging.info(f"Random agent initialized.")
print(f"Random agent initialized.\n")


class QMIXSingleAgentWrapper:
    def __init__(self, qmix_agent):
        self.agent = qmix_agent
    def select_action(self, obs_batch, epsilon=0.0):
        obs_list = [obs_batch[0] for _ in range(self.agent.n_agents)]
        return self.agent.select_actions(obs_list, epsilon=epsilon)
    
class HiMAPPOWrapper:
    def __init__(self, himappo_agent):
        self.agent = himappo_agent
        self.current_goal = 0
    def select_action(self, obs_batch):
        return self.agent.select_actions(obs_batch, goal_ids=[self.current_goal])
        
qmixWrapped = QMIXSingleAgentWrapper(qmix)
himappoWrapped = HiMAPPOWrapper(himappo)
logging.info(f"Agents wrapped for single agent simulation.")

gc.collect() #memory cleanuo
if torch.cuda.is_available():
    torch.cuda.empty_cache() #clear GPU memory
logging.info("Memory cleared.")

agents = [mappo, himappoWrapped, qmixWrapped]
agent_names = ["MAPPO", "HiMAPPO", "QMIX"]

sim = MixedAgentSimulator(
    agents=agents,
    agent_names=agent_names,
    rows=rows,
    cols=cols,
    num_episodes=5000,
    log_interval=25
)
logging.info(f"Simulation initialized with {len(agents)} agents: {agent_names}")

simResults = sim.run(stepsPerEp=25, render=True, output_csv=outputFile) or []
logging.info(f"Simulation completed. Results saved to {outputFile}")

env = CivilisationSimulationMixed(rows=rows, cols=cols)
randomData = []
for episode in range(100):  # Run 100 episodes for random agent
    state = env.reset()
    episode_reward = 0
    for step in range(25):  # 25 steps per episode
        action = random.select_action([state])
        state, reward, done, _ = env.step(action)
        episode_reward += sum(reward) if isinstance(reward, list) else reward
    randomData.append({"agent": "Random", "episode": episode, "reward": episode_reward})

combinedData = simResults + randomData
dfCombined = pd.DataFrame(combinedData)
dfCombined.to_csv(outputFile, index=False)
logging.info(f"Results saved to {outputFile}")
print(f"Results saved to {outputFile}\n")

gc.collect() #memory cleanup post simulation
if torch.cuda.is_available():
    torch.cuda.empty_cache() #clear GPU memory
    
sim.env.renderHeatmap(sPath="logs/final_territory_heatmap.png")
logging.info(f"Heatmap saved to logs/final_territory_heatmap.png")
print(f"Heatmap saved to logs/final_territory_heatmap.png\n")