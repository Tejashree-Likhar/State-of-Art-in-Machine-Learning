"""
06_xgboost_model.py
=====================
XGBoost with Bayesian hyperparameter optimisation (FR15) via scikit-optimize's
BayesSearchCV, and early stopping (patience=50) on a held-out validation
split carved from the end of the training set (never touching the test set).
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Real, Integer

from utils import (
    load_ticker_data, chronological_split, scale_minmax, inverse_scale_target,
    evaluate_predictions, TICKERS, SECTOR_MAP, FEATURE_COLS, TARGET_COL,
    set_all_seeds, RANDOM_SEED, RESULTS_METRICS_DIR, RESULTS_PRED_DIR
)

set_all_seeds()

LOG_PATH = os.path.join(RESULTS_METRICS_DIR, "xgboost_training_log.txt")
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


SEARCH_SPACE = {
    "n_estimators": Integer(50, 500),
    "learning_rate": Real(0.001, 0.3, prior="log-uniform"),
    "max_depth": Integer(3, 10),
    "reg_alpha": Real(1e-3, 10, prior="log-uniform"),
    "reg_lambda": Real(1e-3, 10, prior="log-uniform"),
}
N_ITER = 20  # Bayesian optimisation iterations
EARLY_STOPPING_ROUNDS = 50
VALIDATION_FRACTION = 0.15  # last 15% of training data used as early-stopping validation set

log("=" * 70)
log("XGBOOST MODEL - TRAINING LOG")
log("=" * 70)
log(f"Bayesian search space: {list(SEARCH_SPACE.keys())}, n_iter={N_ITER}")
log(f"Early stopping patience: {EARLY_STOPPING_ROUNDS} rounds")

all_columns = FEATURE_COLS + [TARGET_COL]

# --- Checkpoint / resume support (same pattern as Random Forest script) ---
METRICS_CSV = f"{RESULTS_METRICS_DIR}/XGBoost_metrics.csv"
IMPORTANCE_CSV = f"{RESULTS_METRICS_DIR}/XGBoost_feature_importance.csv"

if os.path.exists(METRICS_CSV):
    all_metrics = pd.read_csv(METRICS_CSV).to_dict("records")
    done_tickers = set(r["Ticker"] for r in all_metrics)
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

    train_scaled, test_scaled, scaler = scale_minmax(train_df, test_df, all_columns)

    X_train_full = train_scaled[FEATURE_COLS].values
    y_train_full = train_scaled[TARGET_COL].values
    X_test = test_scaled[FEATURE_COLS].values

    # Step 1: Bayesian hyperparameter search (time-series CV, train set only)
    tscv = TimeSeriesSplit(n_splits=5)
    opt = BayesSearchCV(
        xgb.XGBRegressor(random_state=RANDOM_SEED, n_jobs=1, verbosity=0),
        SEARCH_SPACE,
        n_iter=N_ITER,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    opt.fit(X_train_full, y_train_full)
    best_params = opt.best_params_
    log(f"Best params (Bayesian search): {dict(best_params)}")

    # Step 2: refit final model with early stopping on a chronological
    # validation split carved from the end of the training data
    n_val = int(len(X_train_full) * VALIDATION_FRACTION)
    X_fit, X_val = X_train_full[:-n_val], X_train_full[-n_val:]
    y_fit, y_val = y_train_full[:-n_val], y_train_full[-n_val:]

    final_model = xgb.XGBRegressor(
        **best_params,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbosity=0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="rmse",
    )
    final_model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    log(f"Best iteration (early stopping): {final_model.best_iteration}")

    y_pred_scaled = final_model.predict(X_test)
    y_pred = inverse_scale_target(y_pred_scaled, scaler, all_columns, TARGET_COL)
    y_test_actual = test_df[TARGET_COL].values

    metrics = evaluate_predictions(y_test_actual, y_pred)
    elapsed = time.time() - t0
    log(f"RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"MAPE={metrics['MAPE']:.2f}%  R2={metrics['R2']:.4f}  "
        f"(trained in {elapsed:.1f}s)")

    all_metrics.append({
        "Model": "XGBoost", "Ticker": ticker, "Sector": SECTOR_MAP[ticker],
        "Best_Params": str(dict(best_params)),
        "Best_Iteration": final_model.best_iteration,
        **metrics, "Train_Time_Sec": elapsed
    })

    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual": y_test_actual,
        "Predicted": y_pred
    })
    pred_df.to_csv(f"{RESULTS_PRED_DIR}/XGBoost_{ticker}_predictions.csv", index=False)

    # Feature importance (FR19)
    importance_row = {"Ticker": ticker, "Sector": SECTOR_MAP[ticker]}
    for feat, imp in zip(FEATURE_COLS, final_model.feature_importances_):
        importance_row[feat] = imp
    feature_importances_all.append(importance_row)

    # Save incrementally after every ticker so progress is never lost
    pd.DataFrame(all_metrics).to_csv(METRICS_CSV, index=False)
    pd.DataFrame(feature_importances_all).to_csv(IMPORTANCE_CSV, index=False)

metrics_df = pd.DataFrame(all_metrics)

log("\n" + "=" * 70)
log("XGBOOST SUMMARY (all tickers)")
log("=" * 70)
log(metrics_df[["Ticker", "Sector", "RMSE", "MAE", "MAPE", "R2", "Train_Time_Sec"]].to_string(index=False))
log(f"\nMean RMSE across all stocks: {metrics_df['RMSE'].mean():.4f}")
log(f"Mean MAPE across all stocks: {metrics_df['MAPE'].mean():.2f}%")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n\nXGBoost training log saved to {LOG_PATH}")
print("XGBoost complete.")
