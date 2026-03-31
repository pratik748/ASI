import numpy as np
from typing import Dict, Any, List, Optional
from world_state import WorldState

class NLPInterpreter:
    """
    Perception Engine (Layer I & II of the TRUTH Architecture).
    Translates raw environment inputs into structured state vectors.
    """
    def __init__(self, labels: Dict[str, int]):
        """
        Initialize the NLP Interpreter with the state space definition.

        :param labels: Mapping from variable names to indices in the state vector.
        """
        self.labels = labels
        self.vector_dim = len(labels)

    def parse_to_world_state(self, raw_input: Dict[str, float]) -> WorldState:
        """
        Perception: Convert raw, unstructured data into a WorldState.

        :param raw_input: A dictionary of numeric variables.
        :return: A structured WorldState object representing Sp.
        """
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        for name, value in raw_input.items():
            if name in self.labels:
                vector[self.labels[name]] = float(value)

        return WorldState(vector, self.labels)

    def generate_target_attractor(self, goal_description: Dict[str, float]) -> WorldState:
        """
        Goal Formulation: Translates a goal into the Target Attractor Vector (St).

        :param goal_description: A dictionary of goal values.
        :return: A structured WorldState object representing St.
        """
        vector = np.zeros(self.vector_dim, dtype=np.float32)
        # Any value not specified in goal_description might be set to
        # a 'don't care' state or inherited from Sp if appropriate.
        # Here we just initialize to zeros and fill provided goals.
        for name, value in goal_description.items():
            if name in self.labels:
                vector[self.labels[name]] = float(value)

        return WorldState(vector, self.labels)

    def mock_llm_call(self, prompt: str) -> Dict[str, Any]:
        """
        Mocks a translation from natural language to a state vector schema.
        In production, this would call an LLM (e.g., GPT-4o) with a JSON schema.
        """
        # For the macroeconomic case study:
        if "market steering" in prompt.lower():
            return {
                "gdp_growth": 0.03,
                "inflation": 0.02,
                "unemployment": 0.05,
                "debt_to_gdp": 0.60,
                "interest_rate": 0.04
            }
        return {}
