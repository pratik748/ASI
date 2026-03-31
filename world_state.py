import numpy as np
from typing import Dict, Any, Optional

class WorldState:
    """
    Unified representation of reality (Layer I & II of the TRUTH Architecture).
    Ensures all reasoning happens on structured state vectors.
    """
    def __init__(self, state_vector: np.ndarray, labels: Optional[Dict[str, int]] = None):
        """
        Initialize the World State.

        :param state_vector: A 1D numpy array representing the state.
        :param labels: A mapping from variable names to indices in the state vector.
        """
        self.vector = state_vector.astype(np.float32)
        self.labels = labels or {}

    def get_value(self, name: str) -> float:
        if name not in self.labels:
            raise ValueError(f"Variable '{name}' not found in WorldState labels.")
        return float(self.vector[self.labels[name]])

    def to_dict(self) -> Dict[str, float]:
        return {name: float(self.vector[idx]) for name, idx in self.labels.items()}

    def __repr__(self):
        return f"WorldState({self.to_dict()})"

    def copy(self):
        return WorldState(self.vector.copy(), self.labels.copy())
