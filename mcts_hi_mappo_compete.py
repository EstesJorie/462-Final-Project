import copy
import math
import random
import torch

# set seed for reproducibility
SEED = 7
random.seed(SEED)
torch.manual_seed(SEED)

class MCTSNode:
    """
    Node in the MCTS tree.

    Each node stores:
    - current environment state
    - the goal taken to reach this node
    - value and visit stats
    - parent and children references
    """
    def __init__(self, state, parent=None, goal=None):
        self.state = state
        self.parent = parent
        self.goal = goal
        self.children = {}
        self.visits = 0
        self.value = 0.0

    def is_expanded(self):
        """Check if all goals have been expanded from this node."""
        return len(self.children) == 3  # assuming goal space is [0, 1, 2]

class MCTS:
    """
    Monte Carlo Tree Search (MCTS) for goal selection in Hi-MAPPO.

    Each simulation explores possible high-level goals for a specific tribe
    and evaluates them using rollout and score accumulation.
    """
    def __init__(self, agent, env, controlled_tribe_id, state_tensor, num_simulations=25, gamma=0.9, c_puct=1.0):
        self.agent = agent
        self.env = env
        self.controlled_tribe_id = controlled_tribe_id  # 1-based tribe ID
        self.state_tensor = state_tensor
        self.num_simulations = num_simulations
        self.gamma = gamma
        self.c_puct = c_puct
        self.goal_space = [0, 1, 2]  # predefined goal IDs

    def run(self):
        """
        Run MCTS and return the best goal based on visit count.
        """
        root = MCTSNode(copy.deepcopy(self.env))

        for _ in range(self.num_simulations):
            node = root
            path = []

            # traverse down to a leaf
            while node.is_expanded():
                node = self.select_child(node)
                path.append(node)

            # expand node if not yet expanded
            if not node.is_expanded():
                self.expand_node(node)

            # simulate rollout to get reward
            reward = self.simulate(node)

            # backpropagate reward
            for n in path:
                n.visits += 1
                n.value += reward

        # select goal with highest visit count from root's children
        best_goal = max(root.children.items(), key=lambda kv: kv[1].visits)[0]
        return best_goal

    def select_child(self, node):
        """
        Select child node using UCT (Upper Confidence Bound).
        """
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
        """
        Expand the node by creating child nodes for all possible goals.
        """
        for goal in self.goal_space:
            child_env = copy.deepcopy(node.state)
            state = torch.tensor(child_env.get_global_state(), dtype=torch.float32)

            num_tribes = child_env.num_tribes
            actions = [0] * num_tribes  # initialize default actions

            # only control the specified tribe, others do nothing
            tribe_index = self.controlled_tribe_id - 1
            obs = torch.tensor(child_env.get_obs(self.controlled_tribe_id), dtype=torch.float32)
            goal_tensor = torch.tensor([goal])
            action, _ = self.agent.select_actions([obs], goal_tensor)
            actions[tribe_index] = action[0]

            # apply 2-step rollout
            for _ in range(2):
                child_env.step(actions)

            node.children[goal] = MCTSNode(child_env, parent=node, goal=goal)

    def simulate(self, node):
        """
        Simulate final result from the node by computing sum of all tribe scores.
        """
        return sum(node.state.compute_final_scores())
