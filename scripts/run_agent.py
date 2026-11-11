
"""
Execute the complete Seoul Bike AutoML agent.

Run:
    python -m scripts.run_agent

Outputs:
    artifacts/agent/experiment_history.csv
    artifacts/agent/best_experiment.json
"""

import json
from pathlib import Path

import pandas as pd

from agent.graph import agent


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "agent"


def main():
    """Run iterative experiments and save the final results."""

    initial_state = {
        "iteration": 0,
        "max_iterations": 6,
        "target_mae": None,
        "experiment_history": [],
        "best_result": None,
        "stop_reason": "",
    }

    # A higher recursion limit allows multiple graph iterations.
    result = agent.invoke(
        initial_state,
        config={"recursion_limit": 50},
    )

    history = result["experiment_history"]
    best = result["best_result"]

    # Save all evaluated model configurations and metrics.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    history_df = pd.DataFrame(history)
    history_df.to_csv(
        OUTPUT_DIR / "experiment_history.csv",
        index=False,
    )

    # Save the best validation experiment as structured JSON.
    with open(
        OUTPUT_DIR / "best_experiment.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(best, file, indent=2)

    print("\n=== FINAL AUTOML RESULTS ===")

    print(
        history_df[
            ["experiment_id", "MAE", "RMSE", "R2"]
        ].round(3).to_string(index=False)
    )

    print("\nBest experiment:", best["experiment_id"])
    print(f"Best validation MAE: {best['MAE']:.3f}")
    print("Experiments completed:", result["iteration"])
    print("Stop reason:", result["stop_reason"])

    print("\nResults saved in artifacts/agent/")
    print("Final test set remains untouched.")


if __name__ == "__main__":
    main()
