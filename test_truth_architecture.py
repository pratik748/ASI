import numpy as np
from world_state import WorldState
from nlp_interpreter import NLPInterpreter
from memory_system import MemorySystem
from world_model import MacroeconomicModel, CausalRuleEngine
from manifold_engine import Action, ManifoldEngine
from selection_core import SelectionCore
from execution_loop import ExecutionLoop

def main():
    # 1. Define the State Space
    labels = {
        "gdp_growth": 0,
        "inflation": 1,
        "unemployment": 2,
        "debt_to_gdp": 3,
        "interest_rate": 4
    }

    # 2. Define Boundary Conditions (The Truth Crucible)
    # Ensuring physical/economic laws: e.g. GDP can't be negative (in growth),
    # interest rates must be >= 0, etc.
    bounds = {
        "gdp_growth": (-0.10, 0.15),
        "inflation": (-0.05, 0.50),
        "unemployment": (0.01, 0.25),
        "debt_to_gdp": (0.0, 3.0),
        "interest_rate": (0.0, 0.20)
    }

    # 3. Initialize TRUTH Architecture Modules
    interpreter = NLPInterpreter(labels)
    memory = MemorySystem(len(labels))
    causal_engine = CausalRuleEngine(bounds)
    model = MacroeconomicModel(labels, causal_engine)
    manifold_engine = ManifoldEngine(model)
    selection_core = SelectionCore(causal_engine, memory)

    loop = ExecutionLoop(interpreter, memory, model, causal_engine, manifold_engine, selection_core)

    # 4. Define Initial Environment State (Sp) and Target Attractor (St)
    # Current environment: recession-like scenario
    raw_input = {
        "gdp_growth": -0.01,
        "inflation": 0.01,
        "unemployment": 0.07,
        "debt_to_gdp": 0.85,
        "interest_rate": 0.01
    }

    # Target state: Stable growth with low inflation
    goal = {
        "gdp_growth": 0.03,
        "inflation": 0.02,
        "unemployment": 0.04,
        "debt_to_gdp": 0.75,
        "interest_rate": 0.04
    }

    # 5. Candidate Actions (Candidate trajectory forking)
    candidate_actions = [
        Action(np.array([0.02, 0.0, -0.01, -0.05, 0.01], dtype=np.float32), name="Stimulus Package"),
        Action(np.array([-0.01, 0.01, 0.0, 0.0, 0.02], dtype=np.float32), name="Rate Hike (Austerity)"),
        Action(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), name="Inaction (Status Quo)"),
        Action(np.array([0.10, 0.20, -0.10, 0.50, -0.05], dtype=np.float32), name="Hyper Stimulus (Risky)"),
    ]

    print("--- Starting TRUTH Architecture Cycle ---")

    # Run multiple cycles to demonstrate self-evolution (Scar Memory build-up)
    for cycle in range(5):
        print(f"\nCycle {cycle + 1}:")
        result = loop.run_cycle(raw_input, goal, candidate_actions)

        print(f"Status: {result.get('status')}")
        if result.get('status') == 'Committed':
            print(f"Action Taken: {result.get('action_name')}")
            print(f"Fitness (SFF): {result.get('best_sff'):.4f}")
            print(f"New World State: {result.get('actual_state')}")
            print(f"Cycle Latency: {result.get('latency'):.4f}s")
            print(f"Scar Memory Total: {result.get('memory_usage')}")

            # Feed the new actual state back for the next cycle
            raw_input = result.get('actual_state')
        else:
            print(f"Reason: {result.get('reason') or result.get('tension')}")

    print("\n--- TRUTH Architecture Cycle Complete ---")

if __name__ == "__main__":
    main()
