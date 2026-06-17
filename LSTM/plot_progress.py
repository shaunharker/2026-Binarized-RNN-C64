# plot_progress.py

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress


def plot_progress(csv_path: str = "data.csv", n: int = 10000) -> None:
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    if df.empty:
        print("Error: CSV file is empty.")
        return

    required_columns = {
        "step",
        "loss_nats_per_token",
        "loss_bits_per_token",
        "unixtime",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        print(
            "Error: CSV file is missing required columns: "
            f"{sorted(missing_columns)}"
        )
        return

    # Convert columns to numeric.
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=list(required_columns))

    if df.empty:
        print("Error: CSV file contains no valid numeric rows.")
        return

    # Calculate time since beginning.
    start_time = df["unixtime"].iloc[0]
    df["time_elapsed"] = df["unixtime"] - start_time

    # Create the plot.
    plt.figure(figsize=(10, 6))

    plt.plot(
        df["time_elapsed"],
        df["loss_bits_per_token"],
        label="Loss",
        color="blue",
        linewidth=2,
    )

    # Filter for the last `n` steps
    max_step = df["step"].max()
    df_fit = df[df["step"] >= (max_step - n)]

    if len(df_fit) > 1:
        # Perform regression of loss against step
        # (Fitting against 'step' is robust to time gaps/pauses in logging)
        res = linregress(df_fit["step"], df_fit["loss_bits_per_token"])

        # Calculate the 1-sided p-value for the hypothesis test:
        # Null Hypothesis (H0): slope >= 0 vs. Alternative (Ha): slope < 0
        if res.slope < 0:
            p_value = res.pvalue / 2
        else:
            p_value = 1.0 - (res.pvalue / 2)

        # Calculate expected change per hour based on timestamps in the window
        time_diff_seconds = df_fit["unixtime"].max() - df_fit["unixtime"].min()
        step_diff = df_fit["step"].max() - df_fit["step"].min()

        if time_diff_seconds > 0 and step_diff > 0:
            steps_per_hour = (step_diff / time_diff_seconds) * 3600
            change_per_hour = res.slope * steps_per_hour
            change_per_hour_str = f"{change_per_hour:.8e} bits/token"
        else:
            change_per_hour_str = "N/A (insufficient time/step range)"

        # Calculate the estimated current value using the regression line at the latest step
        current_step = df_fit["step"].max()
        current_value_estimate = res.slope * current_step + res.intercept

        # Print regression statistics
        print(f"\n--- Regression Fit (Last {n} steps) ---")
        print(f"Data points used: {len(df_fit)}")
        print(f"Slope: {res.slope:.8e} bits/token per step")
        print(f"Expected change per hour: {change_per_hour_str}")
        print(f"Current value estimate (at step {current_step}): {current_value_estimate:.8e} bits/token")
        print(f"p-value (rejecting slope >= 0): {p_value:.8e}")
        print("-----------------------------------------\n")

        # Plot the regression line onto the time_elapsed axis
        fitted_loss = res.slope * df_fit["step"] + res.intercept
        plt.plot(
            df_fit["time_elapsed"],
            fitted_loss,
            label=f"Regression Fit (Last {n} steps)",
            color="red",
            linestyle="--",
            linewidth=2,
        )
    else:
        print(f"\nNot enough data points in the last {n} steps to perform a regression fit.\n")

    plt.title("QAT Algorithm Progress: Loss vs Time", fontsize=14)
    plt.xlabel("Time Elapsed (seconds)", fontsize=12)
    plt.ylabel("Loss (bits/token)", fontsize=12)

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot progress and calculate regression.")
    
    parser.add_argument(
        "csv_path",
        type=str,
        nargs="?",
        default="data.csv",
        help="Path to the CSV file (default: data.csv)",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        default=10000,
        help="Number of last steps to include in the regression fit (default: 10000)",
    )
    
    args = parser.parse_args()

    plot_progress(csv_path=args.csv_path, n=args.n)
