"""
utils.py
========
Shared utilities used by every model script:
  - load_ticker_data(): load a single ticker's feature-engineered CSV
  - chronological_split(): 80/20 time-based train/test split (FR6)
  - scale_minmax() / scale_zscore(): fit-on-train-only scalers (FR4, FR5, NFR4)
  - create_lstm_sequences(): sliding-window sequence builder for LSTM (FR13)
  - evaluate_predictions(): RMSE, MAE, MAPE, R2 (FR16)

IMPORTANT (NFR4 - Data Integrity): every scaler in this file is fit ONLY on
the training partition and then applied (transform only) to the test
partition, to prevent lookahead / data leakage.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------
# CONFIG (shared across all model scripts)
# ------------------------------------------------------------------
# Resolve DATA_DIR relative to this file's location (not the caller's CWD),
# so scripts work whether run from src/ or src/models/
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, "..", "data", "processed")
TRAIN_END_DATE = "2023-08-31"   # Train: Jan 2019(ish) - Aug 2023
TEST_START_DATE = "2023-09-01"  # Test:  Sep 2023 - Dec 2024

TICKERS = ["AAPL", "NVDA", "JNJ", "PFE", "JPM", "GS", "XOM", "CVX", "PG", "KO"]

SECTOR_MAP = {
    "AAPL": "Technology", "NVDA": "Technology",
    "JNJ": "Healthcare", "PFE": "Healthcare",
    "JPM": "Financials", "GS": "Financials",
    "XOM": "Energy", "CVX": "Energy",
    "PG": "Consumer Goods", "KO": "Consumer Goods",
}

# Engineered feature columns used as model inputs (excludes raw OHLCV + Close target)
FEATURE_COLS = [
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_12", "EMA_26", "MACD", "MACD_signal",
    "Close_lag_1", "Close_lag_5", "Close_lag_10",
    "Volume_lag_1", "Volume_lag_5", "Volume_lag_10",
    "RSI_14", "ROC_5", "ROC_10",
    "Daily_Return", "Volatility_10",
]

TARGET_COL = "Close"

RANDOM_SEED = 42  # NFR5 - reproducibility

# Project root and results directories, resolved robustly regardless of CWD
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
RESULTS_METRICS_DIR = os.path.join(PROJECT_ROOT, "results", "metrics")
RESULTS_PRED_DIR = os.path.join(PROJECT_ROOT, "results", "predictions")
RESULTS_FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")
os.makedirs(RESULTS_METRICS_DIR, exist_ok=True)
os.makedirs(RESULTS_PRED_DIR, exist_ok=True)
os.makedirs(RESULTS_FIGURES_DIR, exist_ok=True)


def load_ticker_data(ticker):
    """Load the feature-engineered CSV for a single ticker."""
    path = f"{DATA_DIR}/{ticker}_features.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def chronological_split(df, train_end=TRAIN_END_DATE, test_start=TEST_START_DATE):
    """80/20 chronological split per FR6. No shuffling - order is preserved."""
    train_df = df[df["Date"] <= train_end].reset_index(drop=True)
    test_df = df[df["Date"] >= test_start].reset_index(drop=True)
    return train_df, test_df


def scale_minmax(train_df, test_df, columns):
    """
    Min-Max scale the given columns to [0, 1]. Fit on train only (NFR4).
    Used for tree-based models (RF, XGBoost) and Linear Regression (FR5).
    Returns scaled copies of train/test plus the fitted scaler (for inverse-transform).
    """
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    scaler.fit(train_df[columns])
    train_scaled[columns] = scaler.transform(train_df[columns])
    test_scaled[columns] = scaler.transform(test_df[columns])
    return train_scaled, test_scaled, scaler


def scale_zscore(train_df, test_df, columns):
    """
    Z-score (standard) scale the given columns. Fit on train only (NFR4).
    Used for LSTM input (FR4).
    """
    scaler = StandardScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    scaler.fit(train_df[columns])
    train_scaled[columns] = scaler.transform(train_df[columns])
    test_scaled[columns] = scaler.transform(test_df[columns])
    return train_scaled, test_scaled, scaler


def inverse_scale_target(scaled_values, scaler, columns, target_col=TARGET_COL):
    """
    Inverse-transform a 1D array of scaled target predictions back to
    price-scale (NFR6). Requires reconstructing a dummy full-width array
    since sklearn scalers were fit on multiple columns simultaneously.
    """
    target_idx = columns.index(target_col)
    dummy = np.zeros((len(scaled_values), len(columns)))
    dummy[:, target_idx] = scaled_values
    inv = scaler.inverse_transform(dummy)[:, target_idx]
    return inv


def create_lstm_sequences(data, target, window=60):
    """
    Build sliding-window sequences for LSTM input (FR13).
    data:   2D array (n_samples, n_features) already scaled
    target: 1D array (n_samples,) already scaled (same scaling space as data's Close col)
    window: number of past timesteps used to predict the next value
    Returns X (n_samples-window, window, n_features), y (n_samples-window,)
    """
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(target[i])
    return np.array(X), np.array(y)


def evaluate_predictions(y_true, y_pred):
    """Compute RMSE, MAE, MAPE, R2 on price-scale predictions (FR16)."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def set_all_seeds(seed=RANDOM_SEED):
    """Set random seeds across libraries for reproducibility (NFR5)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
