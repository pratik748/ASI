import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from world_state import WorldState

class CausalRuleEngine:
    """
    The Truth Crucible (Layer II & IV of the TRUTH Architecture).
    Implements matrix-based constraints and invariant boundary conditions.
    """
    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        """
        Initialize with world constraints.

        :param bounds: Dictionary of (min, max) for state variables.
        """
        self.bounds = bounds

    def validate_and_constrain(self, state_vector: np.ndarray, labels: Dict[str, int]) -> np.ndarray:
        """
        Layer IV: Invariant physical/economic laws as boundary conditions.
        Any generated state matrix that violates these bounds must be instantly zeroed out
        (or clamped/nullified) to maintain physical realism.

        :param state_vector: The raw vector to validate.
        :param labels: Mapping of variable names to indices.
        :return: Constrained vector (elements zeroed if they violate bounds).
        """
        constrained_vector = state_vector.copy()
        for name, (min_val, max_val) in self.bounds.items():
            if name in labels:
                idx = labels[name]
                val = constrained_vector[idx]
                if val < min_val or val > max_val:
                    # Instantly zero out the entire state or just the violating component?
                    # The prompt says "generated state matrix that violates these bounds must be instantly zeroed out".
                    # To be rigorous, we return a zero vector to indicate an invalid state manifold.
                    return np.zeros_like(constrained_vector)

        return constrained_vector

    def calculate_entropy_delta(self, state_a: WorldState, state_b: WorldState) -> float:
        """
        Layer VI: Observer Effect.
        Calculates the entropy change between two states to penalize high-chaos transitions.
        Using a more rigorous Frobenius norm of the state transition difference.
        """
        delta = np.linalg.norm(state_a.vector - state_b.vector)
        return float(delta)

class MacroeconomicModel:
    """
    A domain-specific World Model (Causal Engine) for market steering.
    Defines the transition function for the world using matrix-based logic (MFL).
    """
    def __init__(self, labels: Dict[str, int], rule_engine: CausalRuleEngine):
        self.labels = labels
        self.dim = len(labels)
        self.rule_engine = rule_engine

        # Matrix-based transition (MFL Core)
        self.A = np.eye(self.dim, dtype=np.float32)

        # Domain constraints (Okun's Law, Taylor Rule, Debt Dynamics)
        if "gdp_growth" in labels and "unemployment" in labels:
            self.A[labels["unemployment"], labels["gdp_growth"]] = -0.5

        if "inflation" in labels and "interest_rate" in labels:
            self.A[labels["interest_rate"], labels["inflation"]] = 0.5

        if "debt_to_gdp" in labels and "gdp_growth" in labels:
            # High GDP growth reduces debt ratio
            self.A[labels["debt_to_gdp"], labels["gdp_growth"]] = -0.2

    def step_forward(self, state: WorldState, action_vector: np.ndarray, dt: float = 0.1) -> WorldState:
        """
        Layer IV: Temporal Evolution using a symplectic-like integration step.
        """
        # S(t+dt) = S(t) + (A @ S(t) + action) * dt
        rate_of_change = (self.A @ state.vector) + action_vector
        new_vector = state.vector + rate_of_change * dt

        # Apply the Truth Crucible constraints (Zeroing out violations)
        final_vector = self.rule_engine.validate_and_constrain(new_vector, self.labels)

        return WorldState(final_vector, state.labels)
