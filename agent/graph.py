
"""
Complete LangGraph AutoML experimentation workflow.

Workflow:
    Inspector -> Planner -> Trainer -> Evaluator -> Decision

The Decision node controls a conditional loop:
    continue -> Planner
    stop     -> END
"""

from langgraph.graph import START, END, StateGraph

from agent.state import AgentState
from agent.nodes import (
    inspect_data,
    plan_experiment,
    train_model,
    evaluate_experiment,
    make_decision,
    route_after_decision,
)


def build_agent():
    """Build and compile the iterative AutoML agent."""

    workflow = StateGraph(AgentState)

    # Register all five custom workflow nodes.
    workflow.add_node("data_inspector", inspect_data)
    workflow.add_node("experiment_planner", plan_experiment)
    workflow.add_node("model_trainer", train_model)
    workflow.add_node("evaluator", evaluate_experiment)
    workflow.add_node("decision", make_decision)

    # Build the sequential part of the workflow.
    workflow.add_edge(START, "data_inspector")
    workflow.add_edge("data_inspector", "experiment_planner")
    workflow.add_edge("experiment_planner", "model_trainer")
    workflow.add_edge("model_trainer", "evaluator")
    workflow.add_edge("evaluator", "decision")

    # Continue experimenting or terminate the workflow.
    workflow.add_conditional_edges(
        "decision",
        route_after_decision,
        {
            "continue": "experiment_planner",
            "stop": END,
        },
    )

    return workflow.compile()


agent = build_agent()
