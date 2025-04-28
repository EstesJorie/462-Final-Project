import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from civilisation_simulation_env_mixed import CivilisationSimulationMixed

class MixedAgentSimulator:
    def __init__(self, agents, agent_names=None, rows=10, cols=10, num_episodes=5000, log_interval=10, n_runs_per_algo=30):
        """
        Args:
            agents (list): List of trained agent instances (MAPPO, Hi-MAPPO, QMIX, etc.)
            agent_names (list): Optional list of agent names, one per agent.
            rows (int): Grid rows.
            cols (int): Grid columns.
            num_episodes (int): Number of full episodes to simulate.
            log_interval (int): How often to print/render.
            n_runs_per_algo (int): Number of runs per agent algorithm.
        """
        self.agents = agents
        self.agent_names = agent_names or [f"Agent_{i+1}" for i in range(len(agents))]
        self.rows = rows
        self.cols = cols
        self.num_episodes = num_episodes
        self.log_interval = log_interval
        self.n_runs_per_algo = n_runs_per_algo
        self.env = CivilisationSimulationMixed(rows, cols, len(agents), agents)

    def run(self, stepsPerEp=25, render=False, output_csv="sim_mixed_agent_scores.csv", log_actions=False):
        data = []
        for run_id in tqdm(range(1, self.n_runs_per_algo + 1), desc="Running Agent Simulations"):
            self.env.reset()
           
            for ep in range(1, self.num_episodes + 1):
                ep_actions = []
                # Assume 'pop_score', 'food_score', 'territory_score' are calculated at each step
                for _ in range(stepsPerEp):
                    self.env.step()
                    if log_actions:
                        ep_actions.append(self.env.actions_last_step)
               
                pop_score = self.env.get_population_score()  # Example method to retrieve population score
                food_score = self.env.get_food_score()  # Example method to retrieve food score
                territory_score = self.env.get_territory_score()  # Example method to retrieve territory score
                final_score = 0.5 * pop_score + 0.35 * territory_score + 0.15 * food_score

                # Log the results for each episode, turn, and agent
                for agent_idx, agent_name in enumerate(self.agent_names):
                    data.append({
                        'run_id': run_id,
                        'algorithm': agent_name,
                        'turn': ep,  # Assuming 'turn' corresponds to episodes here
                        'episode': ep,
                        'pop_score': pop_score[agent_idx],
                        'food_score': food_score[agent_idx],
                        'territory_score': territory_score[agent_idx],
                        'final_score': final_score[agent_idx]
                    })

            if render and run_id % 10 == 0:
                self.env.render()
                self.env.renderHeatmap(sPath=f"logs/heatmap_run_{run_id}.png")
                print(f"Run {run_id} complete.")

        # Create the output CSV with the desired format
        os.makedirs("logs", exist_ok=True)
        file_path = os.path.join("logs", output_csv)
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=['run_id', 'algorithm', 'turn', 'episode', 'pop_score', 'food_score', 'territory_score', 'final_score'])
            writer.writeheader()
            writer.writerows(data)

        print(f"Simulation results saved to '{file_path}'.")

        # Stats summary (optional)
        print("\nSimulation complete. Here's a quick summary:")
        # You can compute averages or other summaries here if you like

        # Score trend plot (optional)
        plt.figure(figsize=(14, 8))
        for agent_idx, agent_name in enumerate(self.agent_names):
            agent_scores = [entry['final_score'] for entry in data if entry['algorithm'] == agent_name]
            plt.plot(agent_scores, label=agent_name)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Agent Performance Over Episodes")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plot_path = os.path.join("logs", "performance_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance plot saved to '{plot_path}'.")