import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from world_state import WorldState
from nlp_interpreter import NLPInterpreter
from memory_system import MemorySystem
from world_model import MacroeconomicModel, CausalRuleEngine
from manifold_engine import Action, ManifoldEngine
from selection_core import SelectionCore
from execution_loop import ExecutionLoop

# --- Configuration & Styling ---
st.set_page_config(page_title="TRUTH Architecture | Machine God Interface", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #00ff41;
        font-family: 'Courier New', Courier, monospace;
    }
    .stButton>button {
        background-color: #1a1c24;
        color: #00ff41;
        border: 1px solid #00ff41;
    }
    h1, h2, h3 {
        color: #00ff41 !important;
        text-shadow: 0 0 10px #00ff41;
    }
    .stMetric {
        background-color: #1a1c24;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #00ff41;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialization ---
if 'loop' not in st.session_state:
    labels = {
        "gdp_growth": 0, "inflation": 1, "unemployment": 2,
        "debt_to_gdp": 3, "interest_rate": 4
    }
    bounds = {
        "gdp_growth": (-0.10, 0.15), "inflation": (-0.05, 0.50),
        "unemployment": (0.01, 0.25), "debt_to_gdp": (0.0, 3.0),
        "interest_rate": (0.0, 0.20)
    }
    interpreter = NLPInterpreter(labels)
    memory = MemorySystem(len(labels))
    causal_engine = CausalRuleEngine(bounds)
    model = MacroeconomicModel(labels, causal_engine)
    manifold_engine = ManifoldEngine(model)
    selection_core = SelectionCore(causal_engine, memory)

    st.session_state.loop = ExecutionLoop(interpreter, memory, model, causal_engine, manifold_engine, selection_core)
    st.session_state.current_state = {
        "gdp_growth": -0.01, "inflation": 0.01, "unemployment": 0.07,
        "debt_to_gdp": 0.85, "interest_rate": 0.01
    }
    st.session_state.history = []

# --- Sidebar: Control Panel ---
st.sidebar.title("MFL CORE CONTROL")
st.sidebar.subheader("Target Attractor (St)")
goal_gdp = st.sidebar.slider("Target GDP Growth", -0.05, 0.10, 0.03)
goal_inflation = st.sidebar.slider("Target Inflation", 0.0, 0.10, 0.02)
goal_debt = st.sidebar.slider("Target Debt/GDP", 0.10, 1.50, 0.70)

target_attractor = {
    "gdp_growth": goal_gdp, "inflation": goal_inflation,
    "unemployment": 0.04, "debt_to_gdp": goal_debt, "interest_rate": 0.04
}

st.sidebar.subheader("Candidate Interventions")
if st.sidebar.button("EXECUTE REALITY FORK"):
    candidate_actions = [
        Action(np.array([0.02, 0.0, -0.01, -0.05, 0.01], dtype=np.float32), name="Stimulus Package"),
        Action(np.array([-0.01, 0.01, 0.0, 0.0, 0.02], dtype=np.float32), name="Rate Hike"),
        Action(np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), name="Status Quo"),
        Action(np.array([0.10, 0.20, -0.10, 0.50, -0.05], dtype=np.float32), name="Hyper Stimulus")
    ]

    result = st.session_state.loop.run_cycle(st.session_state.current_state, target_attractor, candidate_actions)
    if result['status'] == 'Committed':
        st.session_state.current_state = result['actual_state']
        st.session_state.history.append(result)
    else:
        st.error(f"Cycle Aborted: {result.get('reason', 'Unknown error')}")

# --- Main Dashboard ---
st.title("TRUTH ARCHITECTURE | SYSTEM OBSERVER")

col1, col2, col3, col4 = st.columns(4)
col1.metric("GDP GROWTH", f"{st.session_state.current_state['gdp_growth']:.2%}")
col2.metric("INFLATION", f"{st.session_state.current_state['inflation']:.2%}")
col3.metric("UNEMPLOYMENT", f"{st.session_state.current_state['unemployment']:.2%}")
col4.metric("DEBT / GDP", f"{st.session_state.current_state['debt_to_gdp']:.2f}")

st.divider()

# --- Visualizations ---
if st.session_state.history:
    st.subheader("Temporal Evolution Trajectory")
    df_history = pd.DataFrame([r['actual_state'] for r in st.session_state.history])
    df_history['Cycle'] = range(1, len(df_history) + 1)

    fig = px.line(df_history, x='Cycle', y=['gdp_growth', 'inflation', 'unemployment'],
                  title="Macroeconomic Vector Convergence", template="plotly_dark")
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cognitive Stack Metrics")
    m_col1, m_col2 = st.columns(2)

    # Latency Plot
    latencies = [r['latency'] for r in st.session_state.history]
    fig_lat = px.bar(x=range(len(latencies)), y=latencies, title="Cycle Latency (ms)", template="plotly_dark")
    m_col1.plotly_chart(fig_lat, use_container_width=True)

    # SFF Scores
    sffs = [r['best_sff'] for r in st.session_state.history]
    fig_sff = px.line(x=range(len(sffs)), y=sffs, title="Dominant Manifold Fitness (SFF)", template="plotly_dark")
    m_col2.plotly_chart(fig_sff, use_container_width=True)

    st.subheader("Scar Memory Index (Layer X)")
    st.write(f"Total Archived Transitions: {st.session_state.loop.memory.index.ntotal}")

    # Radar Chart of Current State vs Target
    st.subheader("State Vector Alignment (Sp vs St)")
    categories = list(st.session_state.current_state.keys())
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=[st.session_state.current_state[c] for c in categories],
        theta=categories, fill='toself', name='Perceived Reality (Sp)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[target_attractor[c] for c in categories],
        theta=categories, fill='toself', name='Target Attractor (St)'
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), template="plotly_dark")
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.info("Awaiting System Ignition. Execute Reality Fork to begin.")

st.markdown("---")
st.caption("TRUTH Architecture | Sehwag Doctrine v1.0 | Closed-Loop Decision Engine")
