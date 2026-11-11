
"""
Live LangGraph dashboard for Seoul Bike AutoML.

This Streamlit app visualizes the execution of our
custom five-node agent workflow and displays validation
metrics after each completed experiment.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# Make project modules importable when Streamlit starts.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import agent


NODES = [
    "data_inspector",
    "experiment_planner",
    "model_trainer",
    "evaluator",
    "decision",
]

EDGES = [
    ("data_inspector", "experiment_planner"),
    ("experiment_planner", "model_trainer"),
    ("model_trainer", "evaluator"),
    ("evaluator", "decision"),
    ("decision", "experiment_planner"),
]


def make_graph(completed=None, active=None):
    """Create a graph showing completed and upcoming nodes."""

    completed = completed or set()

    lines = [
        "digraph Agent {",
        'rankdir=LR;',
        'bgcolor="transparent";',
        'node [shape=box, style="rounded,filled", '
        'fontname="Helvetica", fontsize=12];',
    ]

    for node in NODES:
        # Blue indicates the next node; green means completed.
        if node == active:
            color = "#dbeafe"
        elif node in completed:
            color = "#dcfce7"
        else:
            color = "#f1f5f9"

        lines.append(
            f'"{node}" [fillcolor="{color}", '
            f'color="#64748b"];'
        )

    for source, target in EDGES:
        # The backward edge represents the AutoML loop.
        style = "dashed" if source == "decision" else "solid"

        lines.append(
            f'"{source}" -> "{target}" [style={style}];'
        )

    lines.append("}")
    return "\n".join(lines)


def get_next_node(node, changes):
    """Determine the next node for graph highlighting."""

    following = {
        "data_inspector": "experiment_planner",
        "experiment_planner": "model_trainer",
        "model_trainer": "evaluator",
        "evaluator": "decision",
    }

    if node == "decision":
        if changes.get("next_step") == "continue":
            return "experiment_planner"
        return None

    return following.get(node)


# Configure the dashboard layout.
st.set_page_config(
    page_title="Seoul Bike AutoML",
    page_icon="🚲",
    layout="wide",
)

st.title("Seoul Bike AutoML")
st.caption("Live LangGraph Workflow · Agent-Based Machine Learning")

st.write(
    "Train and evaluate regression models using the cleaned "
    "Seoul Bike and Open-Meteo datasets stored in PostgreSQL."
)

# Display the custom graph and execution results.
graph_placeholder = st.empty()

graph_placeholder.graphviz_chart(
    make_graph(active="data_inspector")
)

st.subheader("Experiment Results")
results_placeholder = st.empty()

st.subheader("Agent Execution Log")
log_placeholder = st.empty()

# The button executes our existing LangGraph agent.
if st.button("Run AutoML Agent", type="primary"):

    initial_state = {
        "iteration": 0,
        "max_iterations": 6,
        "target_mae": None,
        "experiment_history": [],
        "best_result": None,
        "stop_reason": "",
    }

    completed = set()
    logs = []
    latest_state = {}

    try:
        # Stream node updates instead of waiting for the
        # entire workflow to finish.
        for update in agent.stream(
            initial_state,
            stream_mode="updates",
            config={"recursion_limit": 50},
        ):

            for node_name, changes in update.items():

                # Update the shared summary of completed nodes.
                latest_state.update(changes)
                completed.add(node_name)

                # Highlight the next node in the graph.
                active = get_next_node(node_name, changes)

                graph_placeholder.graphviz_chart(
                    make_graph(completed, active)
                )

                # Display the actual sequence of node executions.
                logs.append(f"Completed: {node_name}")

                if node_name == "experiment_planner":
                    selected = changes["current_experiment"]["id"]
                    logs.append(f"Selected experiment: {selected}")

                if node_name == "evaluator":
                    history = changes["experiment_history"]

                    results = pd.DataFrame(history)

                    results_placeholder.dataframe(
                        results[
                            ["experiment_id", "MAE", "RMSE", "R2"]
                        ].round(3),
                        hide_index=True,
                    )

                    logs.append(
                        f"Validation MAE: {history[-1]['MAE']:.3f}"
                    )

                if node_name == "decision":
                    logs.append(
                        f"Decision: {changes['next_step']}"
                    )

                log_placeholder.code("\n".join(logs))

        # Present the final result after all graph iterations.
        best = latest_state["best_result"]

        # Save the full experiment history for reproducibility.
        output_dir = PROJECT_ROOT / "artifacts" / "agent"
        output_dir.mkdir(parents=True, exist_ok=True)

        history_df = pd.DataFrame(
            latest_state["experiment_history"]
        )

        history_df.to_csv(
            output_dir / "experiment_history.csv",
            index=False,
        )

        # Save the selected experiment for final holdout evaluation.
        with open(
            output_dir / "best_experiment.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(best, file, indent=2)

        st.success("AutoML workflow completed successfully!")

        col1, col2, col3 = st.columns(3)

        col1.metric("Best Validation MAE", f"{best['MAE']:.3f}")
        col2.metric("Best Validation RMSE", f"{best['RMSE']:.3f}")
        col3.metric("Best Validation R²", f"{best['R2']:.3f}")

        st.write("**Best experiment:**", best["experiment_id"])
        st.write("**Stop reason:**", latest_state["stop_reason"])

    except Exception as error:
        # Show errors directly in the UI for easier debugging.
        st.exception(error)
