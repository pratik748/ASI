import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any, Optional
from world_state import WorldState
from world_model import MacroeconomicModel

class Action:
    """
    Structured representation of a potential intervention (Layer III).
    """
    def __init__(self, vector: np.ndarray, name: str = ""):
        self.vector = vector
        self.name = name

class MCTSNode:
    """
    Monte Carlo Tree Search Node (Layer V: Recursive Foresight).
    """
    def __init__(self, state: WorldState, parent=None, action: Action = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

class ManifoldEngine:
    """
    Parallel Simulation Engine (Layer III & V of the TRUTH Architecture).
    Implements Monte Carlo Tree Search (MCTS) for Layer 5 Recursive Foresight.
    Uses ProcessPoolExecutor for true parallelism.
    """
    def __init__(self, model: MacroeconomicModel, num_workers: int = 8, recursion_decay: float = 0.8):
        """
        Initialize the Manifold Engine.

        :param model: The World Model for trajectory evolution.
        :param num_workers: Number of parallel simulation processes.
        :param recursion_decay: Recursion Decay Constant to prune the search tree.
        """
        self.model = model
        self.num_workers = num_workers
        self.recursion_decay = recursion_decay

    def generate_twins_parallel(self, current_state: WorldState, actions: List[Action], target: WorldState) -> List[Tuple[Action, List[WorldState]]]:
        """
        Fork Sp into parallel trajectory matrices using MCTS.
        Uses ProcessPoolExecutor to bypass the GIL.
        """
        # In a real system, we'd use multiprocess-safe state if needed.
        # Here we perform MCTS for each action candidate in parallel.
        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(self._run_mcts_for_action, current_state, action, target) for action in actions]
            for future in futures:
                results.append(future.result())

        return results

    def _run_mcts_for_action(self, initial_state: WorldState, action: Action, target: WorldState, iterations: int = 50) -> Tuple[Action, List[WorldState]]:
        """
        Implementation of MCTS for a specific action candidate (Layer V).
        """
        # Initialize tree with the first step using the chosen action
        next_state = self.model.step_forward(initial_state, action.vector)
        root = MCTSNode(next_state, action=action)

        for _ in range(iterations):
            # 1. Selection & Expansion
            node = root
            depth = 0
            while node.children:
                # Simple UCB-style selection could be added here
                # Using a greedy selection with recursion decay for now
                node = self._select_best_child(node)
                depth += 1

            # 2. Simulation (Rollout) with Recursion Decay
            # Prune search based on decay constant
            if np.random.rand() < (self.recursion_decay ** depth):
                self._expand_node(node)
                if node.children:
                    node = np.random.choice(node.children)

            # 3. Evaluation (Backpropagation)
            score = self._evaluate_state(node.state, target)
            while node:
                node.visits += 1
                node.value += score
                node = node.parent

        # Return the best trajectory found by the tree search
        best_path = self._get_best_path(root)
        return (action, best_path)

    def _select_best_child(self, node: MCTSNode):
        if not node.children:
            return None
        return max(node.children, key=lambda c: c.value / (c.visits + 1e-6))

    def _expand_node(self, node: MCTSNode):
        # Generate child nodes by slightly perturbing the action space or trying inaction
        # In this macroeconomic case, we test small perturbations
        perturbations = [
            np.zeros_like(node.state.vector), # Inaction
            np.random.normal(0, 0.01, size=node.state.vector.shape) # Small noise
        ]

        for p in perturbations:
            next_state = self.model.step_forward(node.state, p)
            if not np.all(next_state.vector == 0): # Only expand if not zeroed out by rule engine
                child = MCTSNode(next_state, parent=node)
                node.children.append(child)

    def _evaluate_state(self, state: WorldState, target: WorldState) -> float:
        """
        Internal evaluator for MCTS (proximity to target attractor).
        """
        if np.all(state.vector == 0):
            return -1.0 # Penalize zeroed-out invalid states
        proximity = np.linalg.norm(state.vector - target.vector)
        return 1.0 / (1.0 + proximity)

    def _get_best_path(self, root: MCTSNode) -> List[WorldState]:
        path = [root.state]
        curr = root
        while curr.children:
            curr = self._select_best_child(curr)
            path.append(curr.state)
        return path

    def calculate_cost(self, action: Action) -> float:
        return float(np.linalg.norm(action.vector))
