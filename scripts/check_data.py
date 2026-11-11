
"""
Data Validation Script — Seoul Bike AutoML Project

Purpose:
    Inspect the raw Seoul Bike Sharing and Open-Meteo weather datasets
    before data cleaning and transformation.

Data sources:
    1. UCI Seoul Bike Sharing Demand (CSV)
    2. Open-Meteo Historical Weather API (JSON)

Main checks:
    - Required columns and data types
    - Dataset size and date range
    - Missing values and duplicate timestamps
    - Hourly timestamp alignment
    - One-to-one merge coverage

This script is read-only and does not modify source files.
"""

import json
from pathlib import Path

import pandas as pd


# Resolve the project root based on this script's location.
# This makes the script work regardless of the terminal directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Define the external weather variables that will be added to bike data.
WEATHER_FEATURES = [
    "cloud_cover",
    "pressure_msl",
    "apparent_temperature",
]


def load_bike_data():
    """
    Load the original Seoul Bike Sharing dataset.

    Returns:
        pd.DataFrame: Raw bike data with a parsed hourly timestamp.
    """

    # Search recursively in case the ZIP file contains subfolders.
    files = list(RAW_DIR.rglob("SeoulBikeData.csv"))

    if not files:
        raise FileNotFoundError(
            "SeoulBikeData.csv was not found in data/raw."
        )

    # Latin-1 supports the special characters in the UCI CSV.
    df = pd.read_csv(files[0], encoding="latin1")

    # Remove accidental whitespace from column names.
    df.columns = df.columns.str.strip()

    # Ensure essential columns exist before processing.
    required_columns = {
        "Date",
        "Hour",
        "Rented Bike Count",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing bike columns: {missing}")

    # Convert the original day/month/year date into a datetime.
    dates = pd.to_datetime(
        df["Date"],
        format="%d/%m/%Y",
        errors="raise",
    )

    # Convert the hour field to a numeric value.
    hours = pd.to_numeric(df["Hour"], errors="raise")

    # Hours must represent 00:00 through 23:00.
    if not hours.between(0, 23).all():
        raise ValueError("Bike dataset contains invalid hours.")

    if not (hours % 1 == 0).all():
        raise ValueError("Bike dataset contains non-integer hours.")

    # Combine date and hour into one local Seoul timestamp.
    # Both datasets use Seoul local time for the join.
    df["timestamp"] = dates + pd.to_timedelta(
        hours, unit="h"
    )

    return df


def load_weather_data():
    """
    Load the historical weather data from Open-Meteo.

    Returns:
        pd.DataFrame: Weather data with hourly timestamps.
    """

    weather_path = RAW_DIR / "seoul_weather.json"

    if not weather_path.exists():
        raise FileNotFoundError(
            "seoul_weather.json was not found in data/raw."
        )

    # Read the JSON response downloaded from Open-Meteo.
    with open(weather_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    # Verify that the expected hourly data exists.
    if "hourly" not in payload:
        raise ValueError(
            f"Open-Meteo response has no hourly data: {payload}"
        )

    # Confirm that the API response uses Seoul local time.
    if payload.get("timezone") != "Asia/Seoul":
        raise ValueError(
            "Weather data must use the Asia/Seoul timezone."
        )

    # Convert hourly JSON arrays into a DataFrame.
    df = pd.DataFrame(payload["hourly"])

    required_columns = {"time", *WEATHER_FEATURES}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing weather columns: {missing}"
        )

    # Parse the local hourly timestamps.
    df["timestamp"] = pd.to_datetime(
        df["time"],
        errors="raise",
    )

    return df


def inspect_dataset(df, name):
    """
    Print basic quality information about a dataset.

    Args:
        df (pd.DataFrame): Dataset to inspect.
        name (str): Display name of the dataset.
    """

    print(f"\n{'=' * 50}")
    print(f"{name.upper()} DATA QUALITY REPORT")
    print("=" * 50)

    # Display the shape and available columns.
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")

    # Inspect the time range and timestamp uniqueness.
    print(
        "Date range:",
        df["timestamp"].min(),
        "to",
        df["timestamp"].max(),
    )

    print(
        "Duplicate timestamps:",
        df["timestamp"].duplicated().sum(),
    )

    # Print missing-value counts for columns that contain nulls.
    missing_values = df.isna().sum()
    missing_values = missing_values[missing_values > 0]

    print("\nMissing values:")

    if missing_values.empty:
        print("None")
    else:
        print(missing_values)

    # Display Pandas data types to identify conversion issues.
    print("\nData types:")
    print(df.dtypes)


def check_merge(bike, weather):
    """
    Verify that both datasets can be merged by hourly timestamp.

    Args:
        bike (pd.DataFrame): Raw bike dataset.
        weather (pd.DataFrame): Raw weather dataset.

    Returns:
        pd.DataFrame: Temporary merged dataset for validation.
    """

    # Avoid duplicate timestamps before a one-to-one join.
    if bike["timestamp"].duplicated().any():
        raise ValueError("Duplicate bike timestamps detected.")

    if weather["timestamp"].duplicated().any():
        raise ValueError("Duplicate weather timestamps detected.")

    # Keep only the required weather variables for the join.
    weather_subset = weather[
        ["timestamp", *WEATHER_FEATURES]
    ].copy()

    # Preserve all bike records with a LEFT JOIN.
    # validate="one_to_one" prevents accidental row multiplication.
    # indicator=True records whether each row matched.
    merged = bike.merge(
        weather_subset,
        on="timestamp",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    matched = (merged["_merge"] == "both").sum()
    unmatched = (merged["_merge"] == "left_only").sum()

    match_rate = matched / len(bike) * 100

    print(f"\n{'=' * 50}")
    print("DATASET MERGE VALIDATION")
    print("=" * 50)

    print("Bike rows:", len(bike))
    print("Weather rows:", len(weather))
    print("Merged rows:", len(merged))
    print("Matched rows:", matched)
    print("Unmatched rows:", unmatched)
    print(f"Match rate: {match_rate:.2f}%")

    # Check missing values in the newly added weather features.
    print("\nMissing weather values after merge:")
    print(merged[WEATHER_FEATURES].isna().sum())

    # Ensure the join did not lose or multiply bike observations.
    if len(merged) != len(bike):
        raise ValueError("Merge changed the number of bike rows.")

    if unmatched > 0:
        raise ValueError(
            f"{unmatched} bike records have no weather match."
        )

    print("\nMERGE VALIDATION PASSED")

    return merged


def main():
    """Run all raw-data validation checks."""

    # Step 1: Load both independent datasets.
    bike = load_bike_data()
    weather = load_weather_data()

    # Step 2: Inspect each dataset independently.
    inspect_dataset(bike, "Seoul Bike")
    inspect_dataset(weather, "Historical Weather")

    # Step 3: Validate the target variable.
    target = pd.to_numeric(
        bike["Rented Bike Count"],
        errors="coerce",
    )

    if target.isna().any():
        raise ValueError("The target contains missing values.")

    if (target < 0).any():
        raise ValueError("Negative rental counts detected.")

    print("\nTarget validation: PASSED")

    # Step 4: Verify that hourly timestamps match.
    check_merge(bike, weather)

    print("\nALL DATA VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()



