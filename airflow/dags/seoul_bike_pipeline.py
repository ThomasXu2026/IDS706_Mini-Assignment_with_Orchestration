
"""
Seoul Bike AutoML — Airflow ETL Pipeline

This DAG builds a custom data engineering workflow:

1. Clean Seoul Bike Sharing Demand data.
2. Clean Open-Meteo historical weather data.
3. Merge the datasets using hourly timestamps.
4. Load the merged data into PostgreSQL.
5. Verify the stored database records.

The two cleaning tasks can execute in parallel.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task


# Files are shared with Airflow through Docker volumes.
PROJECT_ROOT = Path("/opt/airflow/project")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dag(
    dag_id="seoul_bike_pipeline",
    description="Seoul Bike and Weather ETL Pipeline",
    schedule="0 6 * * *",  # Run daily at 06:00 UTC.
    start_date=datetime(2026, 11, 1),
    catchup=False,
    max_active_runs=1,
    tags=["seoul-bike", "data-engineering", "automl"],
)
def seoul_bike_pipeline():

    @task
    def clean_bike_data():
        """Read, clean, and save the bike demand dataset."""

        # Read the original UCI CSV file.
        df = pd.read_csv(
            RAW_DIR / "SeoulBikeData.csv",
            encoding="latin1",
        )

        # Combine the Date and Hour columns into a single timestamp.
        df["timestamp"] = (
            pd.to_datetime(df["Date"], format="%d/%m/%Y")
            + pd.to_timedelta(df["Hour"], unit="h")
        )

        # Each hourly observation must have a unique timestamp.
        if df["timestamp"].isna().any():
            raise ValueError("Bike data contains invalid timestamps.")

        if df["timestamp"].duplicated().any():
            raise ValueError("Bike data contains duplicate timestamps.")

        if df["Rented Bike Count"].isna().any():
            raise ValueError("Bike data contains missing target values.")

        # Preserve the original feature names and columns.
        output_path = PROCESSED_DIR / "bike_clean.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        print(f"Cleaned bike records: {len(df)}")

        return str(output_path)

    @task
    def clean_weather_data():
        """Read, clean, and save the independent weather dataset."""

        # Read the Open-Meteo JSON file.
        with open(
            RAW_DIR / "seoul_weather.json",
            encoding="utf-8",
        ) as file:
            weather = json.load(file)

        # Build a DataFrame directly from the hourly observations.
        df = pd.DataFrame(weather["hourly"])

        # Convert the weather time field into a common join key.
        df["timestamp"] = pd.to_datetime(df.pop("time"))

        if df["timestamp"].isna().any():
            raise ValueError("Weather data contains invalid timestamps.")

        if df["timestamp"].duplicated().any():
            raise ValueError("Weather data contains duplicate timestamps.")

        output_path = PROCESSED_DIR / "weather_clean.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        print(f"Cleaned weather records: {len(df)}")

        return str(output_path)

    @task
    def merge_datasets(bike_path, weather_path):
        """Merge the two cleaned datasets by timestamp."""

        # Read outputs from the two independent cleaning tasks.
        bike = pd.read_csv(
            bike_path,
            parse_dates=["timestamp"],
        )

        weather = pd.read_csv(
            weather_path,
            parse_dates=["timestamp"],
        )

        # Join the datasets without manually mapping every column.
        merged = pd.merge(
            bike,
            weather,
            on="timestamp",
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        # Ensure every bike observation matches one weather record.
        unmatched = (merged["_merge"] != "both").sum()

        if unmatched > 0:
            raise ValueError(
                f"{unmatched} bike timestamps lack weather records."
            )

        merged = merged.drop(columns=["_merge"])

        # Reject incomplete data before database loading.
        if merged.isna().any().any():
            raise ValueError("Merged dataset contains missing values.")

        output_path = PROCESSED_DIR / "bike_weather_merged.csv"
        merged.to_csv(output_path, index=False)

        print(f"Merged rows: {len(merged)}")
        print(f"Merged columns: {len(merged.columns)}")

        return str(output_path)

    @task
    def load_to_postgres(merged_path):
        """Load the merged DataFrame using our existing SQL loader."""

        # Reuse our previously tested PostgreSQL functions.
        from scripts.load_postgres import (
            get_engine,
            upload_to_postgres,
        )

        df = pd.read_csv(
            merged_path,
            parse_dates=["timestamp"],
        )

        engine = get_engine()

        try:
            # Replace the historical records in one transaction.
            upload_to_postgres(df, engine)
        finally:
            engine.dispose()

        print(f"Uploaded {len(df)} records to PostgreSQL.")

        return len(df)

    @task
    def verify_database_rows(expected_rows):
        """Confirm that PostgreSQL contains the expected records."""

        from sqlalchemy import text
        from scripts.load_postgres import (
            get_engine,
            verify_database,
        )

        engine = get_engine()

        try:
            with engine.connect() as connection:
                stored_rows = connection.execute(
                    text("SELECT COUNT(*) FROM bike_demand_cleaned")
                ).scalar_one()

            if stored_rows != expected_rows:
                raise ValueError(
                    f"Expected {expected_rows} records, "
                    f"but found {stored_rows}."
                )

            # Reuse the SQL verification from our original script.
            verify_database(engine)

            print("Database verification passed.")
        finally:
            engine.dispose()

    @task
    def analyze_bike_demand():
        """
        Run baseline machine learning analysis using PostgreSQL data.

        Compare Random Forest models with original UCI features
        and weather-enhanced features.

        Save validation metrics for the project report.
        """

        # Reuse the baseline ML script developed in Part 2.
        # The script reads the merged data directly from PostgreSQL.
        from scripts.baseline_ml import main as run_baseline

        print("Starting baseline ML analysis...")

        # Train and evaluate both feature configurations.
        run_baseline()

        print("Airflow ML analysis completed successfully.")

    # Both data sources are processed independently.
    bike_file = clean_bike_data()
    weather_file = clean_weather_data()

    # Merge only after both cleaning tasks succeed.
    merged_file = merge_datasets(bike_file, weather_file)

    
    # Load the merged dataset into PostgreSQL.
    uploaded_rows = load_to_postgres(merged_file)

    # Verify that the database contains the expected records.
    verification = verify_database_rows(uploaded_rows)

    # Run ML analysis only after database verification succeeds.
    analysis = analyze_bike_demand()

    # Define the dependency between verification and analysis.
    verification >> analysis



# Register the DAG so Airflow can discover it.
seoul_bike_dag = seoul_bike_pipeline()

