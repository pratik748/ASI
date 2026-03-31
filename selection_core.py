import numpy as np
from typing import List, Tuple, Dict, Optional
from world_state import WorldState
from world_model import CausalRuleEngine
from manifold_engine import Action, ManifoldEngine
from memory_system import MemorySystem

class SelectionCore:
    """
    Scoring & Decision Gate (Layers VI, VII, VIII).
    Implements Tension Ignition and the Sehwag Fitness Function (SFF).
    """
    def __init__(self,
                 causal_engine: CausalRuleEngine,
                 memory: MemorySystem,
                 ignition_threshold: float = 0.05,
                 reflexivity_penalty_weight: float = 0.2):
        """
        Initialize the Selection Core.
        """
        self.causal_engine = causal_engine
        self.memory = memory
        self.ignition_threshold = ignition_threshold
        self.reflexivity_penalty_weight = reflexivity_penalty_weight

    def check_tension_ignition(self, Sp: WorldState, St: WorldState) -> bool:
        """
        Layer VII: Tension Ignition.
        T = ||Sp - St|| * grad U
        """
        delta_s = np.linalg.norm(Sp.vector - St.vector)
        tension = float(delta_s)
        return tension > self.ignition_threshold

    def calculate_sff(self,
                      Sp: WorldState,
                      St: WorldState,
                      action: Action,
                      trajectory: List[WorldState],
                      manifold_engine: ManifoldEngine) -> float:
        """
        Layer VIII: Sehwag Fitness Function (SFF).
        F(m) = E[Goal] / (P(Risk) * Cost)
        """
        final_state = trajectory[-1]

        # 1. Expected Goal Achievement (E[Goal]): Proximity to St.
        if np.all(final_state.vector == 0):
            return 0.0

        proximity = np.linalg.norm(final_state.vector - St.vector)
        expected_goal = 1.0 / (1.0 + proximity)

        # 2. Probability of Risk (P(Risk)):
        risk_penalty = 1.0

        # Check boundary violations along the trajectory
        for s in trajectory:
            if np.all(s.vector == 0):
                risk_penalty += 10.0 # High penalty for path death

            # Layer X: Trajectory-based Scar Memory Pruning.
            # Does any state in this proposed path resemble a past trauma?
            path_scar_penalty = self.memory.get_bias_adjustment(s)
            risk_penalty += path_scar_penalty

        # Reflexivity (Observer Effect - Layer VI)
        reflexivity = self.causal_engine.calculate_entropy_delta(Sp, final_state)
        risk_penalty += self.reflexivity_penalty_weight * reflexivity

        # 3. Cost (Layer VIII)
        cost = 1.0 + manifold_engine.calculate_cost(action)

        # SFF Calculation
        sff_score = expected_goal / (risk_penalty * cost)

        return sff_score

    def select_dominant_trajectory(self,
                                   Sp: WorldState,
                                   St: WorldState,
                                   simulated_manifolds: List[Tuple[Action, List[WorldState]]],
                                   manifold_engine: ManifoldEngine) -> Tuple[Optional[Action], Optional[List[WorldState]], float]:
        """
        Sort and select the manifold with the highest SFF score.
        """
        best_score = -1.0
        best_action = None
        best_trajectory = None

        for action, trajectory in simulated_manifolds:
            score = self.calculate_sff(Sp, St, action, trajectory, manifold_engine)
            if score > best_score:
                best_score = score
                best_action = action
                best_trajectory = trajectory

        # Decision Gate (Layer VI)
        if best_score < 0.02: # Adjusted threshold for trajectory-based scoring
            return None, None, best_score

        return best_action, best_trajectory, best_score
