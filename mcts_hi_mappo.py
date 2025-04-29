import copy
import math
import random
import torch

# set random seed for reproducibility (important for consistent search results)
SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

class MCTSNode:
    """A node in the MCTS tree. Stores env state, parent, goal taken, children, stats."""
    def __init__(self, state, parent=None, goal=None):
        self.state = state
        self.parent = parent
        self.goal = goal
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_expanded(self):
        """Check if this node has expanded all possible goals."""
        return len(self.children) == 3  # assuming goal space [0, 1, 2]

class MCTS:
    """
    Monte Carlo Tree Search for hierarchical decision-making.

    - Each node represents a full environment snapshot
    - Nodes expand by trying different high-level goals
    - Simulations are short rollouts evaluating goals
    """
    def __init__(self, agent, env, state_tensor, num_simulations=25, gamma=0.9, c_puct=1.0):
        self.agent = agent
        self.env = env
        self.state_tensor = state_tensor  # (not directly used in tree)
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.c_puct = c_puct
        self.goal_space = [0, 1, 2]

    def run(self):
        """Main MCTS loop: perform simulations and return the best goal."""
        root = MCTSNode(copy.deepcopy(self.env))

        for _ in range(self.num_simulations):
            node = root
            path = []

            while node.is_expanded():
                node = self.select_child(node)
                path.append(node)

            if not node.is_expanded():
                self.expand_node(node)

            reward = self.simulate(node)

            # backpropagate reward up the path
            for n in path:
                n.visits += 1
                n.value += reward

        # pick goal from root's children based on most visits
        best_goal = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        return best_goal

    def select_child(self, node):
        """Select child with highest UCT score (Q + U)."""
        total_visits = sum(child.visits for child in node.children.values())
        best_score = -float('inf')
        best_child = None

        for goal, child in node.children.items():
            q = child.value / (child.visits + 1e-6)
            u = self.c_puct * math.sqrt(total_visits + 1e-6) / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand_node(self, node):
        """Expand all possible goals from current node."""
        for goal in self.goal_space:
            child_env = copy.deepcopy(node.state)
            state = torch.tensor(child_env.get_global_state(), dtype=torch.float32)
            obs_batch = [torch.tensor(o, dtype=torch.float32) for o in child_env.get_agent_obs()]

            # pick actions conditioned on this hypothetical goal
            actions, _ = self.agent.select_actions(obs_batch, torch.tensor([goal]*self.agent.num_agents))

            # do a short 2-step rollout
            for _ in range(2):
                child_env.step(actions)

            node.children[goal] = MCTSNode(child_env, parent=node, goal=goal)

    def simulate(self, node):
        """Simulate by evaluating final civilization scores at this node."""
        return sum(node.state.compute_final_scores())
