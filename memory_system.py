import numpy as np
import faiss
from typing import List, Tuple, Optional
from world_state import WorldState

class MemorySystem:
    """
    Experience Engine (Layer II & X of the TRUTH Architecture).
    Stores and retrieves experience to modify decision-making via Scar Memory.
    """
    def __init__(self, dimension: int, capacity: int = 100000):
        """
        Initialize the Memory System.

        :param dimension: Vector dimension of World States.
        :param capacity: Max entries in long-term memory.
        """
        self.dimension = dimension
        self.capacity = capacity

        # FAISS Index: Inner Product for similarity after normalization,
        # or L2 distance. Here we use L2 distance for "closeness" to past trauma.
        self.index = faiss.IndexFlatL2(dimension)

        # Metadata storage: (Predicted, Actual, OutcomeScore, IsTrauma)
        # Using a list to hold metadata corresponding to FAISS index entries.
        self.metadata = []

    def archive_experience(self,
                           state: WorldState,
                           predicted: WorldState,
                           actual: WorldState,
                           score: float,
                           is_trauma: bool = False):
        """
        Layer X: Trauma Archiving (Scar Memory).
        Adds a transition to long-term memory.
        """
        # Normalize vector for FAISS if needed, but let's stick with raw L2.
        vector = state.vector.reshape(1, -1)
        self.index.add(vector)
        self.metadata.append({
            "predicted": predicted,
            "actual": actual,
            "score": score,
            "is_trauma": is_trauma
        })

        # Basic eviction policy
        if len(self.metadata) > self.capacity:
            # Note: FAISS IndexFlatL2 doesn't support easy deletion.
            # In a production system, we'd use ID-based removal or reconstruct index.
            pass

    def retrieve_similar_scars(self, state: WorldState, k: int = 5, distance_threshold: float = 0.5) -> List[dict]:
        """
        Layer X: Scar Memory retrieval.
        Finds previous world states mathematically similar to the current state.

        :param state: The current state to check for trauma.
        :param k: Number of similar states to retrieve.
        :param distance_threshold: Maximum L2 distance for a relevant "scar".
        :return: List of metadata entries for nearby traumas.
        """
        if self.index.ntotal == 0:
            return []

        vector = state.vector.reshape(1, -1)
        distances, indices = self.index.search(vector, k)

        scars = []
        for d, i in zip(distances[0], indices[0]):
            if i != -1 and d <= distance_threshold:
                meta = self.metadata[i]
                if meta["is_trauma"]:
                    scars.append(meta)

        return scars

    def get_bias_adjustment(self, state: WorldState) -> float:
        """
        Layer VIII & X: Experience-driven preference formation.
        Calculates a bias adjustment based on nearby Scar Memory.
        Returns a penalty to be applied to the current path's fitness score.
        """
        scars = self.retrieve_similar_scars(state)
        if not scars:
            return 0.0

        # Accumulate trauma penalties based on similarity and severity
        penalty = 0.0
        for scar in scars:
            # Negative scores in past experiences increase the current penalty
            penalty += max(0, -scar["score"])

        return penalty / len(scars)
