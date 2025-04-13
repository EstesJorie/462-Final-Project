from CivilisationSimEnv import CivilizationEnv
import numpy as np
import time

# Create the environment
env = CivilizationEnv(rows=5, cols=5, num_tribes=2, max_steps=10)
obs = env.reset()

print("===== Initial State =====")
env.render()

for step in range(10):
    print(f"\n===== Step {step + 1} =====")

    # Random policy (for testing purposes)
    actions = {}
    for agent_id, agent_obs in obs.items():
        actions[agent_id] = np.random.choice(["gather", "grow", "expand"])

    # Interact with the environment
    obs, rewards, dones, infos = env.step(actions)

    # Print current state
    print("Actions:", actions)
    print("Rewards:", rewards)
    env.render()

    # Check for termination
    if all(dones.values()):
        print("\n✅ Simulation finished: maximum steps reached or all agents done.")
        break

print("\n🎉 Test complete!")
