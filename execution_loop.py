import numpy as np
import time
from typing import List, Tuple, Dict, Any, Optional
from world_state import WorldState
from nlp_interpreter import NLPInterpreter
from memory_system import MemorySystem
from world_model import MacroeconomicModel, CausalRuleEngine
from manifold_engine import Action, ManifoldEngine
from selection_core import SelectionCore

class ExecutionLoop:
    """
    TRUTH Architecture Execution Loop (Layers IX, X, XI).
    Shift reality toward the dominant trajectory and learn from results.
    """
    def __init__(self,
                 interpreter: NLPInterpreter,
                 memory: MemorySystem,
                 model: MacroeconomicModel,
                 causal_engine: CausalRuleEngine,
                 manifold_engine: ManifoldEngine,
                 selection_core: SelectionCore,
                 trauma_threshold: float = 5.0):
        self.interpreter = interpreter
        self.memory = memory
        self.model = model
        self.causal_engine = causal_engine
        self.manifold_engine = manifold_engine
        self.selection_core = selection_core
        self.trauma_threshold = trauma_threshold

    def run_cycle(self, raw_input: Dict[str, float], goal: Dict[str, float], candidate_actions: List[Action]) -> Dict[str, Any]:
        """
        One complete loop of the TRUTH Architecture.
        """
        start_time = time.time()

        # 1. Perception (Layer I & II)
        Sp = self.interpreter.parse_to_world_state(raw_input)
        St = self.interpreter.generate_target_attractor(goal)

        # 2. Tension Ignition (Layer VII)
        if not self.selection_core.check_tension_ignition(Sp, St):
            return {"status": "Quiescent", "tension": "Below Ignition Threshold"}

        # 3. Pre-Simulation Scar Memory Pruning (Layer X)
        # Instantly prune paths if the current state resembles extreme past trauma.
        scar_penalty = self.memory.get_bias_adjustment(Sp)
        if scar_penalty > self.trauma_threshold:
            return {
                "status": "Aborted",
                "reason": f"System Stasis: Current state resembles past trauma (Scar Penalty: {scar_penalty:.2f})"
            }

        # 4. Manifold Twin Generation & MCTS Simulation (Layer V Recursive Foresight)
        # Using ProcessPoolExecutor via generate_twins_parallel
        simulated_manifolds = self.manifold_engine.generate_twins_parallel(Sp, candidate_actions, St)

        # 5. Selection Core (Layer VIII: SFF)
        best_action, best_trajectory, best_score = self.selection_core.select_dominant_trajectory(
            Sp, St, simulated_manifolds, self.manifold_engine
        )

        if not best_action:
            return {"status": "Aborted", "reason": "No safe action found (Decision Gate)"}

        # 6. Reality Commit (Layer IX)
        actual_vector = self.model.step_forward(Sp, best_action.vector).vector
        actual_state = WorldState(actual_vector, Sp.labels)

        # 7. Trauma Archiving & Scar Memory (Layer X)
        manifold_scores = []
        for action, trajectory in simulated_manifolds:
            score = self.selection_core.calculate_sff(Sp, St, action, trajectory, self.manifold_engine)
            manifold_scores.append((action, trajectory, score))

        # Sort by score ascending (lowest SFF = potential trauma)
        manifold_scores.sort(key=lambda x: x[2])

        # Bottom 20%
        num_scars = max(1, len(manifold_scores) // 5)
        for i in range(num_scars):
            action, trajectory, score = manifold_scores[i]
            # Archive the failing state and its predicted outcome
            self.memory.archive_experience(
                state=Sp,
                predicted=trajectory[-1],
                actual=actual_state,
                score=score,
                is_trauma=(score < 0.1) # Mark as trauma if score is critically low
            )

        # 8. Self-Evolution (Layer XI)
        cycle_latency = time.time() - start_time

        return {
            "status": "Committed",
            "action_name": best_action.name,
            "best_sff": best_score,
            "actual_state": actual_state.to_dict(),
            "latency": cycle_latency,
            "memory_usage": self.memory.index.ntotal
        }
