import copy
import math
import random
import torch

# ===============================================
# MCTS Tree Node Class
# Each node represents an environment state resulting from a high-level goal
# ===============================================
class MCTSNode:
    def __init__(self, state, parent=None, goal=None):
        self.state = state              # Deep-copied environment instance at this node
        self.parent = parent            # Pointer to parent node in the tree
        self.goal = goal                # High-level goal that led to this node
        self.children = {}              # Mapping: {goal_id: child_node}
        self.visits = 0                 # Number of times this node has been visited
        self.value = 0.0                # Cumulative reward collected through backpropagation

    # Returns True if the node has been expanded (i.e., has all goal branches)
    def is_expanded(self):
        return len(self.children) == 3  # Assumes goal space is fixed to 3 goals: [0, 1, 2]

# ===============================================
# Monte Carlo Tree Search Implementation
# Used to select high-level goals for Hi-MAPPO using lookahead planning
# ===============================================
class MCTS:
    def __init__(self, agent, env, state_tensor, num_simulations=25, gamma=0.9, c_puct=1.0):
        self.agent = agent                    # HiMAPPOAgent: provides low-level policy
        self.env = env                        # Environment instance (used for copying state)
        self.state_tensor = state_tensor      # Manager's state input (not used directly in tree)
        self.num_simulations = num_simulations
        self.gamma = gamma                    # Discount factor for future rewards
        self.c_puct = c_puct                  # Controls exploration vs exploitation in tree policy
        self.goal_space = [0, 1, 2]           # Available high-level goals

    # Main MCTS loop: performs simulations and returns the best goal
    def run(self):
        root = MCTSNode(copy.deepcopy(self.env))  # Create root node with current environment

        for _ in range(self.num_simulations):
            node = root
            path = []

            # ===== Selection + Expansion =====
            while node.is_expanded():
                node = self.select_child(node)
                path.append(node)

            # ===== Expansion =====
            if not node.is_expanded():
                self.expand_node(node)

            # ===== Simulation (rollout evaluation) =====
            reward = self.simulate(node)

            # ===== Backpropagation =====
            for n in path:
                n.visits += 1
                n.value += reward

        # Choose goal from root with highest number of visits
        best = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        return best

    # ===============================================
    # Select child using UCT (Upper Confidence Bound)
    # Encourages balance between exploitation (Q) and exploration (U)
    # ===============================================
    def select_child(self, node):
        total_visits = sum(child.visits for child in node.children.values())
        best_score = -float('inf')
        best_child = None

        for goal, child in node.children.items():
            q = child.value / (child.visits + 1e-6)  # Estimated value
            u = self.c_puct * math.sqrt(total_visits + 1e-6) / (1 + child.visits)  # Exploration bonus
            score = q + u
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    # ===============================================
    # Expand node by creating all child nodes using possible goals
    # Each child corresponds to one goal in the goal space
    # Simulates 2 steps of environment interaction for each goal
    # ===============================================
    def expand_node(self, node):
        for goal in self.goal_space:
            child_env = copy.deepcopy(node.state)  # Copy environment for this path
            state = torch.tensor(child_env.get_global_state(), dtype=torch.float32)
            obs_batch = [torch.tensor(o, dtype=torch.float32) for o in child_env.get_agent_obs()]

            # Use worker policies to select actions under the given goal
            actions, _ = self.agent.select_actions(obs_batch, torch.tensor([goal]*self.agent.num_agents))

            # Perform 2 rollout steps to evaluate goal effect
            for _ in range(2):
                child_env.step(actions)

            # Create and attach child node to parent
            node.children[goal] = MCTSNode(child_env, parent=node, goal=goal)

    # ===============================================
    # Simulate node by evaluating final civilization scores
    # Returns cumulative score of all agents (sum over tribes)
    # ===============================================
    def simulate(self, node):
        return sum(node.state.compute_final_scores())
