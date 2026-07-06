"""
05_random_forest_model.py
===========================
Random Forest with grid search hyperparameter tuning (FR14).
Grid: n_estimators [100, 200, 500], max_depth [10, 15, 20],
      min_samples_leaf [2, 5, 10]  ->  27 combinations.

Uses TimeSeriesSplit for cross-validation within the training set (never
touching the test set) to select the best configuration by validation RMSE,
in line with NFR4 (no test-set leakage into any fitted parameter/choice).
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from utils import (
    load_ticker_data, chronological_split, scale_minmax, inverse_scale_target,
    evaluate_predictions, TICKERS, SECTOR_MAP, FEATURE_COLS, TARGET_COL,
    set_all_seeds, RANDOM_SEED, RESULTS_METRICS_DIR, RESULTS_PRED_DIR
)

set_all_seeds()

LOG_PATH = os.path.join(RESULTS_METRICS_DIR, "random_forest_training_log.txt")
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [10, 15, 20],
    "min_samples_leaf": [2, 5, 10],
}

log("=" * 70)
log("RANDOM FOREST MODEL - TRAINING LOG")
log("=" * 70)
log(f"Grid search space: {PARAM_GRID}  ({3*3*3} combinations)")

all_columns = FEATURE_COLS + [TARGET_COL]

# --- Checkpoint / resume support ---------------------------------
# If this script is interrupted partway through (e.g. long grid search on
# a slow machine), re-running it will skip tickers already completed by
# reading the existing metrics CSV, rather than starting over from scratch.
METRICS_CSV = f"{RESULTS_METRICS_DIR}/RandomForest_metrics.csv"
IMPORTANCE_CSV = f"{RESULTS_METRICS_DIR}/RandomForest_feature_importance.csv"

if os.path.exists(METRICS_CSV):
    existing_metrics_df = pd.read_csv(METRICS_CSV)
    all_metrics = existing_metrics_df.to_dict("records")
    done_tickers = set(existing_metrics_df["Ticker"])
else:
    all_metrics = []
    done_tickers = set()

if os.path.exists(IMPORTANCE_CSV):
    feature_importances_all = pd.read_csv(IMPORTANCE_CSV).to_dict("records")
else:
    feature_importances_all = []

if done_tickers:
    log(f"\nResuming: found existing results for {sorted(done_tickers)}. Skipping these.")

for ticker in TICKERS:
    if ticker in done_tickers:
        continue
    t0 = time.time()
    log(f"\n--- {ticker} ({SECTOR_MAP[ticker]}) ---")

    df = load_ticker_data(ticker)
    train_df, test_df = chronological_split(df)

    # Min-Max scale (fit on train only)
    train_scaled, test_scaled, scaler = scale_minmax(train_df, test_df, all_columns)

    X_train = train_scaled[FEATURE_COLS].values
    y_train = train_scaled[TARGET_COL].values
    X_test = test_scaled[FEATURE_COLS].values

    # Grid search with time-series-aware CV (no shuffling, respects order)
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
        PARAM_GRID,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    log(f"Best params: {grid.best_params_}")

    y_pred_scaled = best_model.predict(X_test)
    y_pred = inverse_scale_target(y_pred_scaled, scaler, all_columns, TARGET_COL)
    y_test_actual = test_df[TARGET_COL].values

    metrics = evaluate_predictions(y_test_actual, y_pred)
    elapsed = time.time() - t0
    log(f"RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"MAPE={metrics['MAPE']:.2f}%  R2={metrics['R2']:.4f}  "
        f"(trained in {elapsed:.1f}s)")

    all_metrics.append({
        "Model": "RandomForest", "Ticker": ticker, "Sector": SECTOR_MAP[ticker],
        "Best_Params": str(grid.best_params_), **metrics, "Train_Time_Sec": elapsed
    })

    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual": y_test_actual,
        "Predicted": y_pred
    })
    pred_df.to_csv(f"{RESULTS_PRED_DIR}/RandomForest_{ticker}_predictions.csv", index=False)

    # Feature importance (FR19)
    importance_row = {"Ticker": ticker, "Sector": SECTOR_MAP[ticker]}
    for feat, imp in zip(FEATURE_COLS, best_model.feature_importances_):
        importance_row[feat] = imp
    feature_importances_all.append(importance_row)

    # Save incrementally after every ticker so progress is never lost
    pd.DataFrame(all_metrics).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(feature_importances_all).to_csv(IMPORTANCE_CSV, index=False)

metrics_df = pd.DataFrame(all_metrics)

log("\n" + "=" * 70)
log("RANDOM FOREST SUMMARY (all tickers)")
log("=" * 70)
log(metrics_df[["Ticker", "Sector", "RMSE", "MAE", "MAPE", "R2", "Train_Time_Sec"]].to_string(index=False))
log(f"\nMean RMSE across all stocks: {metrics_df['RMSE'].mean():.4f}")
log(f"Mean MAPE across all stocks: {metrics_df['MAPE'].mean():.2f}%")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n\nRandom Forest training log saved to {LOG_PATH}")
print("Random Forest complete.")
