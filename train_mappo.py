import torch
from civilization_env_mappo import CivilizationEnv_MAPPO
from mappo import MAPPOAgent
from GameController import GameController

# 获取用户输入的参数
controller = GameController()
rows, cols = controller.getValidDimensions()
generations = controller.getValidGenerations()
num_tribes = controller.getValidTribeCount()

# 初始化环境和智能体
env = CivilizationEnv_MAPPO(rows=rows, cols=cols, num_tribes=num_tribes)
obs_dim = env.rows * env.cols * 3
act_dim = 3
agent = MAPPOAgent(obs_dim=obs_dim, act_dim=act_dim, num_agents=num_tribes)

# 训练参数
total_iterations = generations
log_interval = 100

for iteration in range(1, total_iterations + 1):
    obs_raw = env.reset()
    trajectories = []

    for step in range(10):
        obs_batch = []
        for agent_id in range(env.num_agents):
            flat_obs = torch.tensor(obs_raw.flatten(), dtype=torch.float32)
            obs_batch.append(flat_obs)

        actions, log_probs = agent.select_action(obs_batch)
        next_obs, rewards, done, _ = env.step(actions)

        trajectories.append({
            'obs': obs_batch,
            'actions': actions,
            'log_probs': log_probs,
            'rewards': rewards
        })

        obs_raw = next_obs

    agent.update(trajectories)

    if iteration % log_interval == 0:
        print(f"\n========== Generation {iteration} ==========")
        env.render()
# 保存模型
import os

save_dir = "trained_models"
os.makedirs(save_dir, exist_ok=True)

for i, actor in enumerate(agent.actors):
    torch.save(actor.state_dict(), os.path.join(save_dir, f"actor_{i}.pth"))
torch.save(agent.critic.state_dict(), os.path.join(save_dir, "critic.pth"))

print("Models saved to 'trained_models/'")
