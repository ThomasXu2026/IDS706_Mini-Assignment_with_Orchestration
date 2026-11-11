
"""
Final holdout evaluation for Seoul Bike AutoML.

The best configuration has already been selected using
validation results from the LangGraph experiment loop.

This script:
1. Reads the selected experiment.
2. Refits that configuration on training + validation data.
3. Evaluates once on the untouched chronological test set.
4. Saves final metrics, predictions, and the refitted model.

The test set must not be used for further model selection.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from scripts.baseline_ml import (
    TARGET,
    create_model,
    get_feature_groups,
    load_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "agent"


def create_selected_regressor(experiment):
    """Rebuild the algorithm selected by the AutoML agent."""

    algorithm = experiment["algorithm"]
    params = experiment["params"]

    if algorithm == "random_forest":
        return RandomForestRegressor(
            **params, random_state=42, n_jobs=-1
        )

    if algorithm == "extra_trees":
        return ExtraTreesRegressor(
            **params, random_state=42, n_jobs=-1
        )

    if algorithm == "gradient_boosting":
        return GradientBoostingRegressor(
            **params, random_state=42
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def main():
    """Refit the selected model and evaluate on held-out data."""

    # Read the winning experiment selected using validation MAE.
    with open(
        OUTPUT_DIR / "best_experiment.json",
        encoding="utf-8",
    ) as file:
        best = json.load(file)

    # Load chronologically sorted records from PostgreSQL.
    df = load_data()

    baseline_features, enhanced_features = get_feature_groups(df)

    if best["feature_set"] == "baseline":
        features = baseline_features
    elif best["feature_set"] == "enhanced":
        features = enhanced_features
    else:
        raise ValueError("Unknown feature set.")

    # Keep the final 15% entirely separate from training.
    split_index = int(len(df) * 0.85)

    train_validation = df.iloc[:split_index]
    test = df.iloc[split_index:]

    X_train = train_validation[features]
    y_train = train_validation[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    # Rebuild the preprocessing pipeline using the selected features.
    model = create_model(X_train)
    regressor = create_selected_regressor(best)

    model.set_params(regressor=regressor)

    # Gradient Boosting requires dense encoded features.
    if best["algorithm"] == "gradient_boosting":
        model.set_params(preprocessor__sparse_threshold=0)

    # Fit using training + validation data only.
    model.fit(X_train, y_train)

    # The test set is used only for final prediction and evaluation.
    predictions = model.predict(X_test)

    metrics = {
        "experiment_id": best["experiment_id"],
        "algorithm": best["algorithm"],
        "feature_set": best["feature_set"],
        "training_rows": len(train_validation),
        "test_rows": len(test),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
        "R2": float(r2_score(y_test, predictions)),
    }

    # Save final evaluation results for the report and dashboard.
    with open(
        OUTPUT_DIR / "final_test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    predictions_df = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "actual": y_test.to_numpy(),
        "predicted": predictions,
    })

    predictions_df.to_csv(
        OUTPUT_DIR / "final_test_predictions.csv",
        index=False,
    )

    # Preserve the model trained on all non-test observations.
    joblib.dump(
        model,
        OUTPUT_DIR / "final_model.joblib",
    )

    print("\n=== FINAL HOLDOUT TEST RESULTS ===")
    print("Selected experiment:", best["experiment_id"])
    print("Training + validation rows:", len(train_validation))
    print("Test rows:", len(test))
    print(f"Test MAE: {metrics['MAE']:.3f}")
    print(f"Test RMSE: {metrics['RMSE']:.3f}")
    print(f"Test R2: {metrics['R2']:.3f}")

    print("\nFinal results and model saved successfully.")


if __name__ == "__main__":
    main()
