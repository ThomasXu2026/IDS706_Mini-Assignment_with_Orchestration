
"""
Seoul Bike AutoML — Data Preparation

Load two independent datasets, clean their timestamp fields,
merge them by hour, and save the processed dataset.
"""

import json
from pathlib import Path

import pandas as pd


# Define paths relative to the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_bike_data():
    """Load and prepare the Seoul Bike Sharing dataset."""

    # Locate and read the original CSV file.
    bike_path = next(RAW_DIR.rglob("SeoulBikeData.csv"))
    bike = pd.read_csv(bike_path, encoding="latin1")

    # Remove extra whitespace from column names.
    bike.columns = bike.columns.str.strip()

    # Convert Date and Hour into a single hourly timestamp.
    bike["timestamp"] = (
        pd.to_datetime(bike["Date"], format="%d/%m/%Y")
        + pd.to_timedelta(bike["Hour"], unit="h")
    )

    # Remove exact duplicate records.
    bike = bike.drop_duplicates()

    # Validate the prediction target.
    if bike["Rented Bike Count"].isna().any():
        raise ValueError("Missing bike rental counts detected.")

    if (bike["Rented Bike Count"] < 0).any():
        raise ValueError("Negative bike rental counts detected.")

    # Ensure timestamps are unique before merging.
    if bike["timestamp"].duplicated().any():
        raise ValueError("Duplicate bike timestamps detected.")

    return bike


def load_weather_data():
    """Load and prepare the Open-Meteo hourly weather dataset."""

    # Read the original JSON response.
    with open(
        RAW_DIR / "seoul_weather.json",
        "r",
        encoding="utf-8",
    ) as file:
        weather_json = json.load(file)

    # Convert hourly weather records into a DataFrame.
    weather = pd.DataFrame(weather_json["hourly"])

    # Convert the weather time column into a timestamp.
    weather["timestamp"] = pd.to_datetime(weather["time"])

    # Remove the redundant time string.
    weather = weather.drop(columns=["time"])

    # Remove exact duplicate records.
    weather = weather.drop_duplicates()

    # Validate timestamp uniqueness.
    if weather["timestamp"].duplicated().any():
        raise ValueError("Duplicate weather timestamps detected.")

    return weather


def merge_data(bike, weather):
    """Merge bike rentals with weather observations by timestamp."""

    # Keep all bike records and add matching weather features.
    merged = pd.merge(
        bike,
        weather,
        on="timestamp",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    # Check whether every bike hour has a weather match.
    unmatched = (merged["_merge"] != "both").sum()

    if unmatched > 0:
        raise ValueError(
            f"{unmatched} bike records have no matching weather data."
        )

    # Remove the temporary merge indicator.
    merged = merged.drop(columns=["_merge"])

    # Sort observations chronologically for future ML experiments.
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    return merged


def main():
    """Run the complete preparation and merge workflow."""

    # Step 1: Load and clean both datasets independently.
    bike = load_bike_data()
    weather = load_weather_data()

    # Step 2: Merge the datasets using the common timestamp.
    merged = merge_data(bike, weather)

    # Step 3: Save the prepared data.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bike.to_csv(OUTPUT_DIR / "bike_clean.csv", index=False)
    weather.to_csv(OUTPUT_DIR / "weather_clean.csv", index=False)
    merged.to_csv(
        OUTPUT_DIR / "bike_weather_merged.csv",
        index=False,
    )

    # Step 4: Report basic data quality and merge results.
    print("\n=== DATA PREPARATION SUMMARY ===")
    print("Bike rows:", len(bike))
    print("Weather rows:", len(weather))
    print("Merged rows:", len(merged))
    print("Merged columns:", len(merged.columns))

    print("\nMissing values:")
    print(merged.isna().sum()[lambda x: x > 0])

    print("\nSample records:")
    print(merged.head())

    print("\nData preparation completed successfully.")


if __name__ == "__main__":
    main()

