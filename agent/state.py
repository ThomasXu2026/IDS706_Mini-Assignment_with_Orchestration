
"""
Shared state for the Seoul Bike AutoML agent.

LangGraph nodes use this state to exchange dataset
information, experiment settings, and evaluation results.

Large DataFrames are not stored directly in the state.
"""

from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # Dataset information collected by the inspector.
    data_summary: dict[str, Any]

    # Feature sets available for model experiments.
    baseline_features: list[str]
    enhanced_features: list[str]

    # Configuration selected for the current experiment.
    current_experiment: dict[str, Any]

    # Results from previous model experiments.
    experiment_history: list[dict[str, Any]]

    # Best experiment found on the validation set.
    best_result: dict[str, Any] | None

    # Track experiment progress and prevent infinite loops.
    iteration: int
    max_iterations: int

    # Explanation for why the workflow stopped.
    stop_reason: str
    # Paths and metadata returned by the training node.
    trained_result: dict[str, Any]
        # Optional validation MAE target for early stopping.
    target_mae: float | None

    # Decision made after each evaluation.
    next_step: str