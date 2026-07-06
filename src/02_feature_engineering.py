"""
02_feature_engineering.py
==========================
Computes technical indicators for each ticker as specified in the
Functional Requirements (FR7-FR10):
  - Simple Moving Averages (5, 10, 20, 50-day)
  - Exponential Moving Averages (12, 26-day) + MACD
  - Lagged Close price and Volume (1, 5, 10 days)
  - RSI (14-day) and Rate of Change (5-day, 10-day)

Run after 01_data_cleaning.py. Reads data/processed/cleaned_combined.csv
and writes data/processed/features_combined.csv plus per-ticker files.
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

INPUT_PATH = "../data/processed/cleaned_combined.csv"
OUTPUT_DIR = "../data/processed"
LOG_PATH = "../results/metrics/feature_engineering_report.txt"

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


def compute_rsi(series, window=14):
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(df_ticker):
    """Compute all technical indicators for a single ticker's time series."""
    df = df_ticker.sort_values("Date").reset_index(drop=True).copy()

    # --- Simple Moving Averages (FR7) ---
    for window in [5, 10, 20, 50]:
        df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()

    # --- Exponential Moving Averages + MACD (FR8) ---
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # --- Lagged Close price and Volume (FR9) ---
    for lag in [1, 5, 10]:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
        df[f"Volume_lag_{lag}"] = df["Volume"].shift(lag)

    # --- RSI and Rate of Change (FR10) ---
    df["RSI_14"] = compute_rsi(df["Close"], window=14)
    df["ROC_5"] = df["Close"].pct_change(periods=5) * 100
    df["ROC_10"] = df["Close"].pct_change(periods=10) * 100

    # --- Additional standard features (daily return, volatility) ---
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(window=10).std()

    return df


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
log("=" * 70)
log("FEATURE ENGINEERING REPORT")
log("=" * 70)

df_all = pd.read_csv(INPUT_PATH)
df_all["Date"] = pd.to_datetime(df_all["Date"])
tickers = sorted(df_all["Ticker"].unique())
log(f"\nLoaded cleaned data: {df_all.shape}")
log(f"Tickers: {tickers}")

results = []
for ticker in tickers:
    sub = df_all[df_all["Ticker"] == ticker]
    feat = engineer_features(sub)
    results.append(feat)
    n_before = len(feat)
    n_after_dropna = feat.dropna().shape[0]
    log(f"\n{ticker}: {n_before} rows total, {n_after_dropna} rows after dropping "
        f"warm-up NaNs (first ~50 days lost to rolling windows)")

df_features = pd.concat(results, ignore_index=True)

# List of engineered feature columns (used later by modelling scripts)
FEATURE_COLS = [
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_12", "EMA_26", "MACD", "MACD_signal",
    "Close_lag_1", "Close_lag_5", "Close_lag_10",
    "Volume_lag_1", "Volume_lag_5", "Volume_lag_10",
    "RSI_14", "ROC_5", "ROC_10",
    "Daily_Return", "Volatility_10",
]

log(f"\nTotal engineered feature columns: {len(FEATURE_COLS)}")
log(str(FEATURE_COLS))

# Drop the warm-up rows containing NaNs from rolling-window computations
# (the max window is 50 days for SMA_50, so ~50 rows per ticker are dropped)
n_before_total = len(df_features)
df_features_clean = df_features.dropna().reset_index(drop=True)
n_after_total = len(df_features_clean)
log(f"\nTotal rows before NaN drop: {n_before_total}")
log(f"Total rows after NaN drop:  {n_after_total}")
log(f"Rows dropped (rolling-window warm-up period): {n_before_total - n_after_total}")

# Save combined feature dataset
combined_out = os.path.join(OUTPUT_DIR, "features_combined.csv")
df_features_clean.to_csv(combined_out, index=False)
log(f"\nSaved: {combined_out}  shape={df_features_clean.shape}")

# Save per-ticker feature files
for ticker in tickers:
    sub = df_features_clean[df_features_clean["Ticker"] == ticker]
    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_features.csv")
    sub.to_csv(out_path, index=False)
    log(f"  Saved {ticker}: {out_path}  ({len(sub)} rows, "
        f"{sub['Date'].min().date()} to {sub['Date'].max().date()})")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n\nFeature engineering report saved to {LOG_PATH}")
print("Feature engineering complete. Proceed to 03_preprocessing.py")
