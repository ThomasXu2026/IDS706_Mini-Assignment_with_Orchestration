
"""
Seoul Bike AutoML — Baseline Model Comparison

Research question:
    Does adding Open-Meteo weather features improve
    hourly bike rental demand prediction?

Models:
    A. UCI original features
    B. UCI features + Open-Meteo weather features

Both models use the same Random Forest settings.

The validation set is used for comparison.
The final test set is reserved for later evaluation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Support running this file directly from the project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.load_postgres import get_engine


TARGET = "Rented Bike Count"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_data():
    """Load the prepared dataset from PostgreSQL."""

    engine = get_engine()

    try:
        with engine.connect() as connection:
            df = pd.read_sql_query(
                text(
                    "SELECT * FROM bike_demand_cleaned "
                    "ORDER BY timestamp"
                ),
                connection,
                parse_dates=["timestamp"],
            )
    finally:
        engine.dispose()

    if df.empty:
        raise ValueError("The database returned no records.")

    if df.isna().any().any():
        raise ValueError("The dataset contains missing values.")

    # Sort chronologically before creating train/validation/test sets.
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Add calendar features available to both models.
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.day_of_week

    return df


def get_feature_groups(df):
    """
    Identify original UCI columns and additional weather columns.

    The UCI schema comes from our existing cleaned CSV,
    avoiding manual specification of every original column.
    """

    bike_schema = pd.read_csv(
        PROCESSED_DIR / "bike_clean.csv",
        nrows=0,
    ).columns.tolist()

    # These fields should not be given directly to the model.
    excluded = {TARGET, "Date", "timestamp"}

    original_features = [
        col for col in bike_schema
        if col not in excluded
    ]

    # Identify features added by the independent weather dataset.
    weather_features = [
        col for col in df.columns
        if col not in bike_schema
        and col not in {"month", "day_of_week"}
    ]

    if not weather_features:
        raise ValueError("No additional weather features found.")

    # Both models receive the same derived calendar features.
    calendar_features = ["month", "day_of_week"]

    baseline_features = original_features + calendar_features
    enhanced_features = baseline_features + weather_features

    print("Additional weather features:", weather_features)

    return baseline_features, enhanced_features


def create_model(X):
    """
    Build a Random Forest pipeline.

    Categorical columns are detected automatically.
    One-hot encoding is fitted using training data only.
    """

    numeric_columns = X.select_dtypes(
        include="number"
    ).columns.tolist()

    categorical_columns = [
        col for col in X.columns
        if col not in numeric_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            ),
            (
                "numeric",
                "passthrough",
                numeric_columns,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def evaluate_model(df, features, model_name):
    """Train a model and evaluate on chronological validation data."""

    # Keep the last 15% untouched for final testing.
    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    train = df.iloc[:train_end]
    validation = df.iloc[train_end:validation_end]

    X_train = train[features]
    y_train = train[TARGET]

    X_validation = validation[features]
    y_validation = validation[TARGET]

    model = create_model(X_train)

    # Fit preprocessing and model using training data only.
    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    return {
        "Model": model_name,
        "MAE": mean_absolute_error(
            y_validation, predictions
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_validation, predictions
            )
        ),
        "R2": r2_score(
            y_validation, predictions
        ),
    }


def main():
    """Compare original and enhanced feature sets."""

    df = load_data()

    baseline_features, enhanced_features = get_feature_groups(df)

    print(f"Database rows: {len(df)}")
    print(f"Baseline features: {len(baseline_features)}")
    print(f"Enhanced features: {len(enhanced_features)}")

    # Train two models with identical algorithms and parameters.
    results = [
        evaluate_model(df, baseline_features, "UCI Baseline"),
        evaluate_model(df, enhanced_features, "UCI + Open-Meteo"),
    ]

    results_df = pd.DataFrame(results)

    print("\n=== VALIDATION RESULTS ===")
    print(results_df.round(3).to_string(index=False))

    # Positive improvement indicates lower prediction error.
    baseline_mae = results_df.loc[0, "MAE"]
    enhanced_mae = results_df.loc[1, "MAE"]

    improvement = (
        (baseline_mae - enhanced_mae)
        / baseline_mae
        * 100
    )

    print(f"\nMAE improvement: {improvement:.2f}%")

    # Save metrics for later use by our LangGraph workflow.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        PROCESSED_DIR / "baseline_results.csv",
        index=False,
    )

    print("Results saved to data/processed/baseline_results.csv")
    print("Final test set remains untouched.")


if __name__ == "__main__":
    main()
