"""
08_evaluate_results.py
========================
Consolidates results from all 5 models into:
  - A single cross-model comparison table (FR18)
  - Sector-level performance summaries (FR21)
  - Prediction vs Actual charts for every model x stock (FR17)
  - Feature importance charts for Random Forest and XGBoost (FR19)
  - A "best model per stock" leaderboard

Run this LAST, after all 5 model scripts have completed.
Outputs go to results/figures/ and results/metrics/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

from utils import TICKERS, SECTOR_MAP, RESULTS_METRICS_DIR, RESULTS_PRED_DIR, RESULTS_FIGURES_DIR

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
})

MODELS = ["Naive", "ARIMA", "LinearRegression", "RandomForest", "XGBoost", "LSTM"]
MODEL_DISPLAY = {
    "Naive": "Naive (Prev. Close)", "ARIMA": "ARIMA", "LinearRegression": "Linear Regression",
    "RandomForest": "Random Forest", "XGBoost": "XGBoost", "LSTM": "LSTM",
}
COLORS = {
    "Naive": "#9467bd", "ARIMA": "#7f7f7f", "LinearRegression": "#1f77b4",
    "RandomForest": "#2ca02c", "XGBoost": "#ff7f0e", "LSTM": "#d62728",
}

# ------------------------------------------------------------------
# STEP 1: LOAD AND COMBINE ALL METRICS
# ------------------------------------------------------------------
all_dfs = []
for model in MODELS:
    path = f"{RESULTS_METRICS_DIR}/{model}_metrics.csv"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping {model}")
        continue
    df = pd.read_csv(path)
    df["Model"] = MODEL_DISPLAY[model]
    all_dfs.append(df[["Model", "Ticker", "Sector", "RMSE", "MAE", "MAPE", "R2"]])

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv(f"{RESULTS_METRICS_DIR}/all_models_combined_metrics.csv", index=False)
print(f"Combined metrics saved. Shape: {combined.shape}")

# ------------------------------------------------------------------
# STEP 2: CROSS-MODEL COMPARISON TABLE (FR18) - pivoted, one per metric
# ------------------------------------------------------------------
for metric in ["RMSE", "MAE", "MAPE", "R2"]:
    pivot = combined.pivot(index="Ticker", columns="Model", values=metric)
    pivot = pivot[[MODEL_DISPLAY[m] for m in MODELS]]  # consistent column order
    pivot = pivot.reindex(TICKERS)  # consistent row order (grouped by sector)
    pivot.to_csv(f"{RESULTS_METRICS_DIR}/comparison_table_{metric}.csv")
    print(f"\n=== {metric} comparison table ===")
    print(pivot.round(3).to_string())

# ------------------------------------------------------------------
# STEP 3: SECTOR-LEVEL PERFORMANCE SUMMARY (FR21)
# ------------------------------------------------------------------
sector_summary = combined.groupby(["Sector", "Model"])[["RMSE", "MAE", "MAPE", "R2"]].mean().reset_index()
sector_summary.to_csv(f"{RESULTS_METRICS_DIR}/sector_level_summary.csv", index=False)
print("\n=== Sector-level mean performance ===")
print(sector_summary.round(3).to_string(index=False))

# ------------------------------------------------------------------
# STEP 4: BEST MODEL PER STOCK (by RMSE and by MAPE)
# ------------------------------------------------------------------
best_by_rmse = combined.loc[combined.groupby("Ticker")["RMSE"].idxmin()][["Ticker", "Sector", "Model", "RMSE", "MAPE", "R2"]]
best_by_rmse = best_by_rmse.set_index("Ticker").reindex(TICKERS).reset_index()
best_by_rmse.to_csv(f"{RESULTS_METRICS_DIR}/best_model_per_stock.csv", index=False)
print("\n=== Best model per stock (lowest RMSE) ===")
print(best_by_rmse.round(3).to_string(index=False))

# Overall mean performance per model (across all 10 stocks)
overall_summary = combined.groupby("Model")[["RMSE", "MAE", "MAPE", "R2"]].mean().reindex(
    [MODEL_DISPLAY[m] for m in MODELS]
)
overall_summary.to_csv(f"{RESULTS_METRICS_DIR}/overall_model_summary.csv")
print("\n=== Overall mean performance per model (across 10 stocks) ===")
print(overall_summary.round(3).to_string())

# ------------------------------------------------------------------
# STEP 5: BAR CHART - RMSE comparison across all models/stocks
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(TICKERS))
width = 0.13
for i, model in enumerate(MODELS):
    vals = [combined[(combined.Ticker == t) & (combined.Model == MODEL_DISPLAY[model])]["RMSE"].values[0]
            if len(combined[(combined.Ticker == t) & (combined.Model == MODEL_DISPLAY[model])]) > 0 else np.nan
            for t in TICKERS]
    ax.bar(x + i * width, vals, width, label=MODEL_DISPLAY[model], color=COLORS[model])
ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
ax.set_xticklabels(TICKERS)
ax.set_ylabel("RMSE (price units, lower is better)")
ax.set_title("RMSE Comparison Across All Models and Stocks")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_FIGURES_DIR}/rmse_comparison_all_models.png", dpi=150)
plt.close()
print(f"\nSaved: rmse_comparison_all_models.png")

# ------------------------------------------------------------------
# STEP 6: BAR CHART - MAPE comparison (percentage, comparable across stocks)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
for i, model in enumerate(MODELS):
    vals = [combined[(combined.Ticker == t) & (combined.Model == MODEL_DISPLAY[model])]["MAPE"].values[0]
            if len(combined[(combined.Ticker == t) & (combined.Model == MODEL_DISPLAY[model])]) > 0 else np.nan
            for t in TICKERS]
    ax.bar(x + i * width, vals, width, label=MODEL_DISPLAY[model], color=COLORS[model])
ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
ax.set_xticklabels(TICKERS)
ax.set_ylabel("MAPE (%, lower is better)")
ax.set_title("MAPE Comparison Across All Models and Stocks")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_FIGURES_DIR}/mape_comparison_all_models.png", dpi=150)
plt.close()
print(f"Saved: mape_comparison_all_models.png")

# ------------------------------------------------------------------
# STEP 7: BAR CHART - R2 comparison (highlighting negative R2 issue)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6))
for i, model in enumerate(MODELS):
    vals = [combined[(combined.Ticker == t) & (combined.Model == MODEL_DISPLAY[model])]["R2"].values[0]
            if len(combined[(combined.Ticker == t) & (combined.Model == MODEL_DISPLAY[model])]) > 0 else np.nan
            for t in TICKERS]
    ax.bar(x + i * width, vals, width, label=MODEL_DISPLAY[model], color=COLORS[model])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
ax.set_xticklabels(TICKERS)
ax.set_ylabel("R² (higher is better; negative = worse than predicting the mean)")
ax.set_title("R² Comparison Across All Models and Stocks")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_FIGURES_DIR}/r2_comparison_all_models.png", dpi=150)
plt.close()
print(f"Saved: r2_comparison_all_models.png")

# ------------------------------------------------------------------
# STEP 8: PREDICTION VS ACTUAL CHARTS - one grid per stock (5 models)
# ------------------------------------------------------------------
for ticker in TICKERS:
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(12, 3.2 * len(MODELS)), sharex=True)
    for i, model in enumerate(MODELS):
        pred_path = f"{RESULTS_PRED_DIR}/{model}_{ticker}_predictions.csv"
        ax = axes[i]
        if not os.path.exists(pred_path):
            ax.set_title(f"{MODEL_DISPLAY[model]} - no data")
            continue
        pred_df = pd.read_csv(pred_path)
        pred_df["Date"] = pd.to_datetime(pred_df["Date"])
        ax.plot(pred_df["Date"], pred_df["Actual"], label="Actual", color="black", linewidth=1.3)
        ax.plot(pred_df["Date"], pred_df["Predicted"], label="Predicted",
                color=COLORS[model], linewidth=1.1, alpha=0.85)
        r2_val = combined[(combined.Ticker == ticker) & (combined.Model == MODEL_DISPLAY[model])]["R2"]
        r2_str = f"{r2_val.values[0]:.3f}" if len(r2_val) > 0 else "N/A"
        ax.set_title(f"{MODEL_DISPLAY[model]}  (R² = {r2_str})", fontsize=11)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{ticker} ({SECTOR_MAP[ticker]}) — Predicted vs Actual Close Price, All Models",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FIGURES_DIR}/predictions_{ticker}_all_models.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: predictions_{ticker}_all_models.png")

# ------------------------------------------------------------------
# STEP 9: FEATURE IMPORTANCE CHARTS (Random Forest & XGBoost) - FR19
# ------------------------------------------------------------------
for model_name, file_prefix in [("RandomForest", "RandomForest"), ("XGBoost", "XGBoost")]:
    imp_path = f"{RESULTS_METRICS_DIR}/{file_prefix}_feature_importance.csv"
    if not os.path.exists(imp_path):
        continue
    imp_df = pd.read_csv(imp_path)
    feature_cols = [c for c in imp_df.columns if c not in ["Ticker", "Sector"]]
    mean_importance = imp_df[feature_cols].mean().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(mean_importance.index, mean_importance.values, color=COLORS.get(model_name, "#333333"))
    ax.set_xlabel("Mean Feature Importance (averaged across all 10 stocks)")
    ax.set_title(f"{MODEL_DISPLAY.get(model_name, model_name)} — Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_FIGURES_DIR}/feature_importance_{model_name}.png", dpi=150)
    plt.close()
    print(f"Saved: feature_importance_{model_name}.png")

# ------------------------------------------------------------------
# STEP 10: SECTOR-LEVEL BAR CHART (mean RMSE by sector and model)
# ------------------------------------------------------------------
sectors = sorted(set(SECTOR_MAP.values()))
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(sectors))
for i, model in enumerate(MODELS):
    vals = [sector_summary[(sector_summary.Sector == s) & (sector_summary.Model == MODEL_DISPLAY[model])]["MAPE"].values
            for s in sectors]
    vals = [v[0] if len(v) > 0 else np.nan for v in vals]
    ax.bar(x + i * width, vals, width, label=MODEL_DISPLAY[model], color=COLORS[model])
ax.set_xticks(x + width * (len(MODELS) - 1) / 2)
ax.set_xticklabels(sectors)
ax.set_ylabel("Mean MAPE (%)")
ax.set_title("Mean MAPE by Sector and Model")
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_FIGURES_DIR}/sector_mape_comparison.png", dpi=150)
plt.close()
print("Saved: sector_mape_comparison.png")

print("\n\nEvaluation complete. All tables saved to results/metrics/, all charts saved to results/figures/")
