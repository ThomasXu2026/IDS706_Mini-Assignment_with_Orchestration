
"""
LangGraph nodes for Seoul Bike AutoML.

Each node receives the current AgentState and returns
only the fields that need to be updated.
"""
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scripts.baseline_ml import create_model
from agent.state import AgentState
from scripts.baseline_ml import (
    TARGET,
    get_feature_groups,
    load_data,
)


def inspect_data(state: AgentState) -> dict:
    """
    Inspect the PostgreSQL dataset and identify feature groups.

    Returns:
        A dictionary containing the dataset summary and
        available model feature sets.
    """

    print("\n=== DATA INSPECTOR ===")

    # Reuse the existing PostgreSQL data loader.
    df = load_data()

    # Identify the original and weather-enhanced features.
    baseline_features, enhanced_features = get_feature_groups(df)

    # Keep only lightweight metadata in the shared state.
    summary = {
        "rows": int(len(df)),
        "start_date": str(df["timestamp"].min()),
        "end_date": str(df["timestamp"].max()),
        "target": TARGET,
        "missing_values": int(df.isna().sum().sum()),
        "baseline_feature_count": len(baseline_features),
        "enhanced_feature_count": len(enhanced_features),
    }

    print("Dataset rows:", summary["rows"])
    print("Target:", summary["target"])
    print("Missing values:", summary["missing_values"])
    print("Baseline features:", len(baseline_features))
    print("Enhanced features:", len(enhanced_features))

    return {
        "data_summary": summary,
        "baseline_features": baseline_features,
        "enhanced_features": enhanced_features,
    }

# Define a reproducible search space for AutoML experiments.
# Each experiment has a unique ID, model, feature set, and parameters.
EXPERIMENTS = [
    {
        "id": "rf_uci",
        "algorithm": "random_forest",
        "feature_set": "baseline",
        "params": {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_leaf": 3,
        },
    },
    {
        "id": "rf_weather",
        "algorithm": "random_forest",
        "feature_set": "enhanced",
        "params": {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_leaf": 3,
        },
    },
    {
        "id": "extra_trees_uci",
        "algorithm": "extra_trees",
        "feature_set": "baseline",
        "params": {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_leaf": 3,
        },
    },
    {
        "id": "extra_trees_weather",
        "algorithm": "extra_trees",
        "feature_set": "enhanced",
        "params": {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_leaf": 3,
        },
    },
    {
        "id": "gradient_boosting_uci",
        "algorithm": "gradient_boosting",
        "feature_set": "baseline",
        "params": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 3,
        },
    },
    {
        "id": "gradient_boosting_weather",
        "algorithm": "gradient_boosting",
        "feature_set": "enhanced",
        "params": {
            "n_estimators": 150,
            "learning_rate": 0.05,
            "max_depth": 3,
        },
    },
]



def plan_experiment(state: AgentState) -> dict:
    """
    Select an untested experiment using previous evaluation results.

    The first two experiments compare original and enhanced
    features using the same Random Forest configuration.

    Later experiments prioritize the feature set associated
    with the best validation MAE so far.
    """

    print("\n=== EXPERIMENT PLANNER ===")

    history = state.get("experiment_history", [])

    # Identify experiments that have already been evaluated.
    attempted_ids = {
        result["experiment_id"]
        for result in history
    }

    remaining = [
        experiment
        for experiment in EXPERIMENTS
        if experiment["id"] not in attempted_ids
    ]

    if not remaining:
        raise ValueError("No untested experiments remain.")

    if len(history) < 2:
        # First establish a fair UCI vs. weather comparison.
        selected = remaining[0]

    else:
        # Use validation feedback to prioritize feature groups.
        best_feature_set = state["best_result"]["feature_set"]

        preferred = [
            experiment
            for experiment in remaining
            if experiment["feature_set"] == best_feature_set
        ]

        # Try the preferred group first, then other experiments.
        selected = preferred[0] if preferred else remaining[0]

    print("Experiment:", selected["id"])
    print("Algorithm:", selected["algorithm"])
    print("Feature set:", selected["feature_set"])
    print("Parameters:", selected["params"])

    return {"current_experiment": selected}


def train_model(state: AgentState) -> dict:
    """
    Train the experiment selected by the planner.

    The dataset is split chronologically:
        70% training
        15% validation
        15% untouched final test

    The trained model and validation predictions are saved
    outside LangGraph State to avoid storing large objects.
    """

    print("\n=== MODEL TRAINER ===")

    experiment = state["current_experiment"]

    # Reload data from PostgreSQL using our existing function.
    df = load_data()

    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]

    # Select features identified by the Data Inspector.
    if experiment["feature_set"] == "baseline":
        features = state["baseline_features"]
    else:
        features = state["enhanced_features"]

    X_train = train[features]
    y_train = train[TARGET]

    X_validation = validation[features]
    y_validation = validation[TARGET]

    # Reuse the preprocessing pipeline from baseline_ml.py.
    model = create_model(X_train)

    algorithm = experiment["algorithm"]
    params = experiment["params"]

    # Instantiate the regressor selected by the planner.
    if algorithm == "random_forest":
        regressor = RandomForestRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
        )

    elif algorithm == "extra_trees":
        regressor = ExtraTreesRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
        )

    elif algorithm == "gradient_boosting":
        regressor = GradientBoostingRegressor(
            **params,
            random_state=42,
        )

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Replace the original Random Forest with the selected model.
    model.set_params(regressor=regressor)

    # Ensure dense encoded features for Gradient Boosting.
    if algorithm == "gradient_boosting":
        model.set_params(
            preprocessor__sparse_threshold=0
        )

    # Fit preprocessing and the model on training data only.
    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    # Store artifacts separately from the agent's shared state.
    project_root = Path(__file__).resolve().parents[1]
    artifact_dir = project_root / "artifacts" / "agent"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    experiment_id = experiment["id"]

    model_path = artifact_dir / f"{experiment_id}.joblib"
    prediction_path = (
        artifact_dir / f"{experiment_id}_validation.csv"
    )

    # Save the fitted model for future evaluation or reuse.
    joblib.dump(model, model_path)

    # Save predictions so the Evaluator can compute metrics.
    validation_results = pd.DataFrame({
        "timestamp": validation["timestamp"].to_numpy(),
        "actual": y_validation.to_numpy(),
        "predicted": predictions,
    })

    validation_results.to_csv(prediction_path, index=False)

    print("Training rows:", len(train))
    print("Validation rows:", len(validation))
    print("Features used:", len(features))
    print("Trained model:", experiment_id)
    print("Model saved:", model_path)
    print("Predictions saved:", prediction_path)

    return {
        "trained_result": {
            "experiment_id": experiment_id,
            "model_path": str(model_path),
            "prediction_path": str(prediction_path),
            "training_rows": len(train),
            "validation_rows": len(validation),
            "feature_count": len(features),
        }
    }

def evaluate_experiment(state: AgentState) -> dict:
    """
    Evaluate the trained model using validation predictions.

    Track every experiment and update the best model based
    on the lowest validation MAE.

    The final test set is not used during this process.
    """

    print("\n=== MODEL EVALUATOR ===")

    experiment = state["current_experiment"]
    trained = state["trained_result"]

    # Load predictions generated by the Model Trainer.
    results = pd.read_csv(trained["prediction_path"])

    required_columns = {"actual", "predicted"}

    if not required_columns.issubset(results.columns):
        raise ValueError("Validation predictions are incomplete.")

    if results.empty:
        raise ValueError("Validation predictions are empty.")

    actual = results["actual"].to_numpy()
    predicted = results["predicted"].to_numpy()

    # Reject invalid predictions before computing metrics.
    if not np.isfinite(actual).all():
        raise ValueError("Actual values contain invalid numbers.")

    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain invalid numbers.")

    # Compute regression performance metrics.
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted))

    # Store the complete configuration for reproducibility.
    current_result = {
        "experiment_id": experiment["id"],
        "algorithm": experiment["algorithm"],
        "feature_set": experiment["feature_set"],
        "params": experiment["params"],
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "model_path": trained["model_path"],
    }

    # Append the result to the experiment history.
    history = state.get("experiment_history", []) + [
        current_result
    ]

    # Compare the new experiment with the historical best.
    previous_best = state.get("best_result")

    if (
        previous_best is None
        or mae < previous_best["MAE"]
    ):
        best_result = current_result
        print("New best model found!")

    else:
        best_result = previous_best
        print("Previous best model retained.")

    print(f"Experiment: {experiment['id']}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"Best MAE so far: {best_result['MAE']:.3f}")

    return {
        "experiment_history": history,
        "best_result": best_result,
        "iteration": len(history),
    }


def make_decision(state: AgentState) -> dict:
    """
    Decide whether another experiment should run.

    Stop if:
        1. The maximum number of experiments is reached.
        2. The optional target MAE is achieved.
        3. No untested configurations remain.

    Otherwise, return to the Experiment Planner.
    """

    print("\n=== AGENT DECISION ===")

    iteration = state["iteration"]
    max_iterations = state["max_iterations"]
    best = state["best_result"]

    target_mae = state.get("target_mae")

    attempted_ids = {
        result["experiment_id"]
        for result in state["experiment_history"]
    }

    if iteration >= max_iterations:
        decision = "stop"
        reason = "Maximum experiment budget reached."

    elif target_mae is not None and best["MAE"] <= target_mae:
        decision = "stop"
        reason = "Target validation MAE achieved."

    elif all(
        experiment["id"] in attempted_ids
        for experiment in EXPERIMENTS
    ):
        decision = "stop"
        reason = "All candidate experiments evaluated."

    else:
        decision = "continue"
        reason = ""

    print(f"Completed experiments: {iteration}/{max_iterations}")
    print("Decision:", decision)

    if reason:
        print("Reason:", reason)

    return {
        "next_step": decision,
        "stop_reason": reason,
    }


def route_after_decision(state: AgentState) -> str:
    """
    Return the next route selected by the Decision node.

    LangGraph maps this result to the Planner or END.
    """

    return state["next_step"]
