"""
09_naive_baseline_and_directional_accuracy.py
================================================
Two additions that strengthen the evaluation chapter without requiring any
model retraining:

1. NAIVE "PREVIOUS CLOSE" BASELINE
   Predicts tomorrow's close = today's actual close. This is the standard
   trivial baseline used by Jayanth (2025) in the literature review. Comparing
   every model against it answers the key critical question: are ARIMA and
   Linear Regression actually learning something, or just approximating a
   random walk? If a model can't beat this baseline, that is a meaningful
   (and honestly reportable) limitation.

2. DIRECTIONAL ACCURACY
   For each model, the percentage of test days where the model correctly
   predicted the DIRECTION of price movement (up/down) relative to the
   previous actual close, regardless of the exact price error. This is a
   standard secondary metric in the financial ML literature (used by
   Jayanth 2025, referenced in the lit review) and is highly relevant for
   trading applications, where getting the direction right matters more than
   the exact price.

Run this AFTER all 5 models have been trained (needs their prediction CSVs)
and BEFORE re-running 08_evaluate_results.py (so the comparison tables and
charts include the naive baseline and directional accuracy).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from utils import (
    load_ticker_data, chronological_split, evaluate_predictions,
    TICKERS, SECTOR_MAP, RESULTS_METRICS_DIR, RESULTS_PRED_DIR
)

MODELS_WITH_PREDICTIONS = ["ARIMA", "LinearRegression", "RandomForest", "XGBoost", "LSTM"]

LOG_PATH = os.path.join(RESULTS_METRICS_DIR, "naive_baseline_and_directional_accuracy_log.txt")
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


log("=" * 70)
log("NAIVE BASELINE + DIRECTIONAL ACCURACY - LOG")
log("=" * 70)

# ------------------------------------------------------------------
# PART 1: NAIVE "PREVIOUS CLOSE" BASELINE
# ------------------------------------------------------------------
log("\n--- Part 1: Naive Previous-Close Baseline ---")

naive_metrics = []

for ticker in TICKERS:
    df = load_ticker_data(ticker)
    train_df, test_df = chronological_split(df)

    # Naive prediction for test day t = actual close at t-1.
    # For the FIRST test day, t-1 is the LAST training day (need this from train_df).
    last_train_close = train_df["Close"].iloc[-1]
    actual_test = test_df["Close"].values
    naive_pred = np.concatenate([[last_train_close], actual_test[:-1]])

    metrics = evaluate_predictions(actual_test, naive_pred)
    log(f"{ticker} ({SECTOR_MAP[ticker]}): RMSE={metrics['RMSE']:.4f}  "
        f"MAE={metrics['MAE']:.4f}  MAPE={metrics['MAPE']:.2f}%  R2={metrics['R2']:.4f}")

    naive_metrics.append({
        "Model": "Naive", "Ticker": ticker, "Sector": SECTOR_MAP[ticker], **metrics
    })

    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual": actual_test,
        "Predicted": naive_pred
    })
    pred_df.to_csv(f"{RESULTS_PRED_DIR}/Naive_{ticker}_predictions.csv", index=False)

naive_df = pd.DataFrame(naive_metrics)
naive_df.to_csv(f"{RESULTS_METRICS_DIR}/Naive_metrics.csv", index=False)

log(f"\nNaive baseline mean RMSE: {naive_df['RMSE'].mean():.4f}")
log(f"Naive baseline mean MAPE: {naive_df['MAPE'].mean():.2f}%")
log(f"Naive baseline mean R2:   {naive_df['R2'].mean():.4f}")

# ------------------------------------------------------------------
# PART 2: DIRECTIONAL ACCURACY (all models, including Naive)
# ------------------------------------------------------------------
log("\n--- Part 2: Directional Accuracy ---")
log("(% of test days where predicted direction of movement matched actual direction,")
log(" measured relative to the previous day's actual close)")

ALL_MODELS = MODELS_WITH_PREDICTIONS + ["Naive"]
directional_results = []

for model in ALL_MODELS:
    for ticker in TICKERS:
        pred_path = f"{RESULTS_PRED_DIR}/{model}_{ticker}_predictions.csv"
        if not os.path.exists(pred_path):
            continue
        pred_df = pd.read_csv(pred_path)

        # Need the actual close from the day BEFORE each test day as the
        # reference point for "direction". First reference = last train close.
        df = load_ticker_data(ticker)
        train_df, test_df = chronological_split(df)
        last_train_close = train_df["Close"].iloc[-1]
        prev_actual = np.concatenate([[last_train_close], pred_df["Actual"].values[:-1]])

        actual_direction = np.sign(pred_df["Actual"].values - prev_actual)
        pred_direction = np.sign(pred_df["Predicted"].values - prev_actual)

        # Treat "no movement" (0) as its own case; only count clear matches
        correct = (actual_direction == pred_direction) & (actual_direction != 0)
        total = (actual_direction != 0).sum()
        directional_accuracy = correct.sum() / total * 100 if total > 0 else np.nan

        directional_results.append({
            "Model": model, "Ticker": ticker, "Sector": SECTOR_MAP[ticker],
            "Directional_Accuracy_Pct": directional_accuracy
        })

dir_df = pd.DataFrame(directional_results)
dir_df.to_csv(f"{RESULTS_METRICS_DIR}/directional_accuracy.csv", index=False)

log("\nMean directional accuracy per model (across all 10 stocks):")
model_display = {
    "ARIMA": "ARIMA", "LinearRegression": "Linear Regression",
    "RandomForest": "Random Forest", "XGBoost": "XGBoost",
    "LSTM": "LSTM", "Naive": "Naive (Previous Close)"
}
summary = dir_df.groupby("Model")["Directional_Accuracy_Pct"].mean().reindex(ALL_MODELS)
for model, val in summary.items():
    log(f"  {model_display[model]:28s}: {val:.2f}%")

summary.to_csv(f"{RESULTS_METRICS_DIR}/directional_accuracy_summary.csv")

# Pivot table: Ticker x Model
pivot = dir_df.pivot(index="Ticker", columns="Model", values="Directional_Accuracy_Pct")
pivot = pivot[[m for m in ALL_MODELS if m in pivot.columns]]
pivot = pivot.reindex(TICKERS)
pivot.to_csv(f"{RESULTS_METRICS_DIR}/directional_accuracy_by_stock.csv")
log("\nDirectional accuracy by stock (%):")
log(pivot.round(1).to_string())

# ------------------------------------------------------------------
# PART 3: CHART - Directional accuracy comparison
# ------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_FIGURES_DIR = os.path.join(os.path.dirname(RESULTS_METRICS_DIR), "figures")
os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)

COLORS = {
    "ARIMA": "#7f7f7f", "LinearRegression": "#1f77b4",
    "RandomForest": "#2ca02c", "XGBoost": "#ff7f0e", "LSTM": "#d62728",
    "Naive": "#9467bd",
}

fig, ax = plt.subplots(figsize=(10, 6))
models_order = ALL_MODELS
vals = [summary[m] for m in models_order]
colors = [COLORS[m] for m in models_order]
bars = ax.bar([model_display[m] for m in models_order], vals, color=colors)
ax.axhline(50, color="black", linestyle="--", linewidth=1, label="Random guessing (50%)")
ax.set_ylabel("Mean Directional Accuracy (%)")
ax.set_title("Directional Accuracy by Model (mean across 10 stocks)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}%",
            ha="center", fontsize=9)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{RESULTS_FIGURES_DIR}/directional_accuracy_comparison.png", dpi=150)
plt.close()
log("\nSaved: directional_accuracy_comparison.png")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n\nLog saved to {LOG_PATH}")
print("Naive baseline + directional accuracy complete.")
