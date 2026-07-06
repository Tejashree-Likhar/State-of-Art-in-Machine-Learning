"""
03_arima_model.py
==================
ARIMA baseline model (FR11). Univariate — uses only the Close price series
(ARIMA cannot natively accept the engineered feature set).

Steps per ticker:
  1. Augmented Dickey-Fuller (ADF) test to determine differencing order d
  2. Grid search over (p, q) combinations, selecting the model with lowest AIC
  3. Rolling one-step-ahead forecasting on the test set (model state updated
     with each true observation, not refit from scratch, for tractability)
  4. Evaluate with RMSE, MAE, MAPE, R2 on price-scale predictions

NOTE ON METHODOLOGY: the dissertation proposal specifies an AIC grid search
over p, q in [0, 5]. This was scoped down to [0, 3] here for computational
tractability across 10 stocks within the project deadline, while preserving
the AIC-minimisation selection principle. This is disclosed as a methodological
adjustment in Chapter 3 of the report.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from utils import (
    load_ticker_data, chronological_split, evaluate_predictions,
    TICKERS, SECTOR_MAP, set_all_seeds,
    RESULTS_METRICS_DIR, RESULTS_PRED_DIR
)

set_all_seeds()

P_RANGE = range(0, 4)   # 0 to 3
Q_RANGE = range(0, 4)   # 0 to 3
MAX_D = 2

LOG_PATH = os.path.join(RESULTS_METRICS_DIR, "arima_training_log.txt")

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


def determine_d(series, max_d=MAX_D, significance=0.05):
    """Use ADF test to find minimum differencing order for stationarity."""
    d = 0
    s = series.copy()
    while d <= max_d:
        result = adfuller(s.dropna())
        p_value = result[1]
        if p_value < significance:
            return d, p_value
        s = s.diff()
        d += 1
    return d, p_value


def grid_search_arima(train_series, d):
    """Grid search (p, q) minimising AIC, with fixed d from ADF test."""
    best_aic = np.inf
    best_order = None
    best_model = None
    for p in P_RANGE:
        for q in Q_RANGE:
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(train_series, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
                    best_model = fitted
            except Exception:
                continue
    return best_order, best_aic, best_model


def rolling_forecast(fitted_model, test_series):
    """
    Rolling one-step-ahead forecast: predict next value, then update the
    model's state with the true observed value (refit=False -> fast Kalman
    filter state update, not a full re-estimation).

    NOTE: test_series must have an integer index that continues directly
    from the training series' index (e.g. train=0..N-1, test=N..N+M-1),
    otherwise statsmodels' `.append()` raises an index-continuity error.
    """
    predictions = []
    current_model = fitted_model
    for i in range(len(test_series)):
        pred = current_model.forecast(steps=1)
        predictions.append(pred.iloc[0] if hasattr(pred, "iloc") else pred[0])
        # update state with the true value at this step
        new_obs = test_series.iloc[[i]]
        current_model = current_model.append(new_obs, refit=False)
    return np.array(predictions)


# ------------------------------------------------------------------
# MAIN LOOP OVER TICKERS
# ------------------------------------------------------------------
log("=" * 70)
log("ARIMA MODEL - TRAINING LOG")
log("=" * 70)

all_metrics = []

for ticker in TICKERS:
    t0 = time.time()
    log(f"\n--- {ticker} ({SECTOR_MAP[ticker]}) ---")

    df = load_ticker_data(ticker)
    train_df, test_df = chronological_split(df)

    # Use a continuing integer index (not DatetimeIndex) so that
    # statsmodels' rolling `.append()` works without frequency errors.
    train_close = train_df["Close"].reset_index(drop=True)
    n_train = len(train_close)
    test_close = test_df["Close"].reset_index(drop=True)
    test_close.index = range(n_train, n_train + len(test_close))

    # Step 1: ADF stationarity test -> determine d
    d, p_value = determine_d(train_close)
    log(f"ADF test: differencing order d={d} (final p-value={p_value:.4f})")

    # Step 2: Grid search p, q by AIC
    order, aic, fitted_model = grid_search_arima(train_close, d)
    log(f"Best order (p,d,q)={order}, AIC={aic:.2f}")

    if fitted_model is None:
        log(f"WARNING: ARIMA failed to converge for {ticker}. Skipping.")
        continue

    # Step 3: Rolling one-step-ahead forecast on test set
    preds = rolling_forecast(fitted_model, test_close)

    # Step 4: Evaluate
    metrics = evaluate_predictions(test_close.values, preds)
    elapsed = time.time() - t0
    log(f"RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"MAPE={metrics['MAPE']:.2f}%  R2={metrics['R2']:.4f}  "
        f"(trained in {elapsed:.1f}s)")

    metrics_row = {
        "Model": "ARIMA", "Ticker": ticker, "Sector": SECTOR_MAP[ticker],
        "Order": str(order), **metrics, "Train_Time_Sec": elapsed
    }
    all_metrics.append(metrics_row)

    # Save predictions
    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual": test_close.values,
        "Predicted": preds
    })
    pred_df.to_csv(f"{RESULTS_PRED_DIR}/ARIMA_{ticker}_predictions.csv", index=False)

# ------------------------------------------------------------------
# SAVE METRICS
# ------------------------------------------------------------------
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(f"{RESULTS_METRICS_DIR}/ARIMA_metrics.csv", index=False)

log("\n" + "=" * 70)
log("ARIMA SUMMARY (all tickers)")
log("=" * 70)
log(metrics_df.to_string(index=False))
log(f"\nMean RMSE across all stocks: {metrics_df['RMSE'].mean():.4f}")
log(f"Mean MAPE across all stocks: {metrics_df['MAPE'].mean():.2f}%")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n\nARIMA training log saved to {LOG_PATH}")
print("ARIMA complete.")
