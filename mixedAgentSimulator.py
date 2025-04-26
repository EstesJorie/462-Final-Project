import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from civilisation_simulation_env_mixed import CivilisationSimulationMixed


class MixedAgentSimulator:
    def __init__(self, agents, agent_names=None, rows=10, cols=10, num_episodes=5000, log_interval=10):
        """
        Args:
            agents (list): List of trained agent instances (MAPPO, Hi-MAPPO, QMIX, etc.)
            agent_names (list): Optional list of agent names, one per agent.
            rows (int): Grid rows.
            cols (int): Grid columns.
            num_episodes (int): Number of full episodes to simulate.
            log_interval (int): How often to print/render.
        """
        self.agents = agents
        self.agent_names = agent_names or [f"Agent_{i+1}" for i in range(len(agents))]
        self.rows = rows
        self.cols = cols
        self.num_episodes = num_episodes
        self.log_interval = log_interval
        self.env = CivilisationSimulationMixed(rows, cols, len(agents), agents) 

    def run(self, stepsPerEp=25, render=False, output_csv="sim_mixed__agent_scores.csv", log_actions=False):
        scoreLog = []
        actionLog = []  

        for ep in tqdm(range(1, self.num_episodes + 1), desc="Running Competition"):
            self.env.reset()
            ep_actions = []

            for _ in range(stepsPerEp):
                self.env.step()
                if log_actions:
                    ep_actions.append(self.env.actions_last_step)  

            finalScores = self.env.compute_final_scores()
            scoreLog.append(finalScores)

            if log_actions:
                actionLog.append(ep_actions)

            if render and ep % self.log_interval == 0:
                self.env.render()
                self.env.renderHeatmap(sPath=f"logs/heatmap_ep_{ep}.png")
                print(f"Episode {ep}: Final Scores: {finalScores}")

        os.makedirs("logs", exist_ok=True)

        # Save scores
        score_path = os.path.join("logs", output_csv)
        with open(score_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode"] + self.agent_names)
            for i, scores in enumerate(scoreLog):
                writer.writerow([i + 1] + scores)

        # Save actions if enabled
        if log_actions:
            action_path = os.path.join("logs", "actions_log.csv")
            with open(action_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Step", "Tribe_ID", "Action"])
                for ep_idx, ep_data in enumerate(actionLog):
                    for step_idx, step_actions in enumerate(ep_data):
                        for tribe_id, action in step_actions:
                            writer.writerow([ep_idx + 1, step_idx + 1, tribe_id, action])
            print(f"Action log saved to '{action_path}'.")

        # Stats summary
        scoreArray = np.array(scoreLog)
        avgScore = np.mean(scoreArray, axis=0)
        bestScore = np.max(scoreArray, axis=0)
        worstScore = np.min(scoreArray, axis=0)

        print("\nFinal Summary:")
        for i, name in enumerate(self.agent_names):
            print(f"{name}: Avg: {avgScore[i]:.2f}, Best: {bestScore[i]:.2f}, Worst: {worstScore[i]:.2f}")

        # Score trend plot
        plt.figure(figsize=(14, 8))
        for i, name in enumerate(self.agent_names):
            plt.plot(scoreArray[:, i], label=name)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Mixed Agent Performance")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plot_path = os.path.join("logs", "performance_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance plot saved to '{plot_path}'.")

        # Final heatmap
        self.env.renderHeatmap(sPath="logs/final_territory_heatmap.png")
        print("\nMixed competition complete.")
        print(f"Scores saved to: {score_path}")
