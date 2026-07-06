"""
04_linear_regression_model.py
==============================
Linear Regression baseline (FR12), directly following AlQahtani et al. (2025)
methodology but extended with the full engineered feature set (SMA, EMA,
MACD, lags, RSI, ROC) rather than raw price alone.

Uses Min-Max scaled features (FR5), fit on training data only (NFR4).
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression

from utils import (
    load_ticker_data, chronological_split, scale_minmax, inverse_scale_target,
    evaluate_predictions, TICKERS, SECTOR_MAP, FEATURE_COLS, TARGET_COL,
    set_all_seeds, RESULTS_METRICS_DIR, RESULTS_PRED_DIR
)

set_all_seeds()

LOG_PATH = os.path.join(RESULTS_METRICS_DIR, "linear_regression_training_log.txt")
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


log("=" * 70)
log("LINEAR REGRESSION MODEL - TRAINING LOG")
log("=" * 70)

all_metrics = []
all_columns = FEATURE_COLS + [TARGET_COL]  # scale features + target together

for ticker in TICKERS:
    t0 = time.time()
    log(f"\n--- {ticker} ({SECTOR_MAP[ticker]}) ---")

    df = load_ticker_data(ticker)
    train_df, test_df = chronological_split(df)

    # Min-Max scale features + target together (fit on train only, NFR4)
    train_scaled, test_scaled, scaler = scale_minmax(train_df, test_df, all_columns)

    X_train = train_scaled[FEATURE_COLS].values
    y_train = train_scaled[TARGET_COL].values
    X_test = test_scaled[FEATURE_COLS].values
    y_test_scaled = test_scaled[TARGET_COL].values

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_scaled = model.predict(X_test)

    # Inverse-transform back to price scale (NFR6)
    y_pred = inverse_scale_target(y_pred_scaled, scaler, all_columns, TARGET_COL)
    y_test_actual = test_df[TARGET_COL].values

    metrics = evaluate_predictions(y_test_actual, y_pred)
    elapsed = time.time() - t0
    log(f"RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"MAPE={metrics['MAPE']:.2f}%  R2={metrics['R2']:.4f}  "
        f"(trained in {elapsed:.2f}s)")

    all_metrics.append({
        "Model": "LinearRegression", "Ticker": ticker, "Sector": SECTOR_MAP[ticker],
        **metrics, "Train_Time_Sec": elapsed
    })

    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual": y_test_actual,
        "Predicted": y_pred
    })
    pred_df.to_csv(f"{RESULTS_PRED_DIR}/LinearRegression_{ticker}_predictions.csv", index=False)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(f"{RESULTS_METRICS_DIR}/LinearRegression_metrics.csv", index=False)

log("\n" + "=" * 70)
log("LINEAR REGRESSION SUMMARY (all tickers)")
log("=" * 70)
log(metrics_df.to_string(index=False))
log(f"\nMean RMSE across all stocks: {metrics_df['RMSE'].mean():.4f}")
log(f"Mean MAPE across all stocks: {metrics_df['MAPE'].mean():.2f}%")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n\nLinear Regression training log saved to {LOG_PATH}")
print("Linear Regression complete.")
