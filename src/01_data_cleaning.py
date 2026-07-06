"""
01_data_cleaning.py
====================
Loads the raw Kaggle datasets, extracts the 10 target stocks across 5 sectors,
cleans the data (duplicates, missing values, formatting, outlier checks),
and saves per-ticker cleaned CSVs plus a combined cleaned dataset.

Data Sources:
- Primary: S&P 500 OHLCV Historical Data (Kaggle, jacksaleeby/s-and-p500-historical-data)
  Originally sourced via yfinance from Yahoo Finance, covering 2000-2026.
- Secondary (evaluated, NOT used): 2019-2024 US Stock Market Data (Kaggle,
  saketk511/2019-2024-us-stock-market-data) - excluded because it only contains
  2 of our 10 required tickers (Apple, Nvidia) and is structured around
  commodities/macro indicators rather than the 5-sector equity set required
  by this dissertation. This decision is documented in Chapter 3.

Run this script first. Outputs go to data/processed/
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
RAW_DATA_PATH = "../data/raw/00_raw_extract_10stocks.csv"
# NOTE: This is a pre-filtered extract (10 tickers, 2019-2024) taken from the full
# S&P 500 dataset (2.7M rows / 472 tickers / 137MB - too large to ship in this
# package). If you want to regenerate this extract yourself from scratch, download
# the full dataset from https://www.kaggle.com/datasets/jacksaleeby/s-and-p500-historical-data,
# place it at ../data/raw/SP500_Historical_Data.csv, and uncomment FULL_RAW_PATH below.
FULL_RAW_PATH = "../data/raw/SP500_Historical_Data.csv"  # optional, not required
OUTPUT_DIR = "../data/processed"
LOG_PATH = "../results/metrics/data_cleaning_report.txt"

START_DATE = "2019-01-01"
END_DATE = "2024-12-31"

# 10 stocks across 5 sectors (as defined in the dissertation proposal)
SECTOR_MAP = {
    "AAPL": "Technology",
    "NVDA": "Technology",
    "JNJ": "Healthcare",
    "PFE": "Healthcare",
    "JPM": "Financials",
    "GS": "Financials",
    "XOM": "Energy",
    "CVX": "Energy",
    "PG": "Consumer Goods",
    "KO": "Consumer Goods",
}
TICKERS = list(SECTOR_MAP.keys())

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


# ------------------------------------------------------------------
# STEP 1: LOAD RAW DATA
# ------------------------------------------------------------------
log("=" * 70)
log("DATA CLEANING REPORT")
log("=" * 70)
log(f"\nLoading raw dataset...")

if os.path.exists(FULL_RAW_PATH):
    log(f"Found full raw dataset at: {FULL_RAW_PATH}")
    df_raw = pd.read_csv(FULL_RAW_PATH)
    log(f"Raw dataset shape (all 472 S&P 500 tickers, 2000-2026): {df_raw.shape}")
    log(f"Raw columns: {list(df_raw.columns)}")

    df = df_raw[df_raw["Ticker"].isin(TICKERS)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    log(f"\nFiltered to {len(TICKERS)} tickers, {START_DATE} to {END_DATE}")
    log(f"Filtered shape: {df.shape}")
else:
    log(f"Full raw dataset not found locally (137MB, excluded from this package).")
    log(f"Loading pre-filtered extract instead from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    log(f"Loaded pre-filtered extract shape: {df.shape}")

# Save the filtered "raw" extract (pre-cleaning) for transparency/reproducibility
df.sort_values(["Ticker", "Date"]).to_csv(
    os.path.join(OUTPUT_DIR, "00_raw_extract_10stocks.csv"), index=False
)

# ------------------------------------------------------------------
# STEP 3: DUPLICATE CHECK
# ------------------------------------------------------------------
n_dupes = df.duplicated(subset=["Ticker", "Date"]).sum()
log(f"\nDuplicate (Ticker, Date) rows found: {n_dupes}")
if n_dupes > 0:
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="first")
    log(f"Duplicates removed. New shape: {df.shape}")
else:
    log("No duplicates found - no action needed.")

# ------------------------------------------------------------------
# STEP 4: MISSING VALUE CHECK
# ------------------------------------------------------------------
log(f"\nMissing values per column (before handling):")
log(str(df.isnull().sum()))

if df.isnull().sum().sum() > 0:
    log("\nApplying forward-fill (per ticker) for missing values...")
    df = df.sort_values(["Ticker", "Date"])
    df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]] = (
        df.groupby("Ticker")[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        .ffill()
    )
    remaining = df.isnull().sum().sum()
    log(f"Remaining missing values after forward-fill: {remaining}")
else:
    log("No missing values found - no imputation needed.")

# ------------------------------------------------------------------
# STEP 5: MISSING TRADING DAY CHECK (calendar completeness per ticker)
# ------------------------------------------------------------------
log("\nTrading day count per ticker (sanity check vs NYSE calendar ~252/year):")
counts = df.groupby("Ticker")["Date"].nunique()
log(str(counts))

# Check that all tickers share an identical trading calendar
date_sets = df.groupby("Ticker")["Date"].apply(lambda x: frozenset(x))
all_identical = date_sets.nunique() == 1
log(f"\nAll tickers share an identical trading calendar: {all_identical}")
if not all_identical:
    # Align all tickers to the intersection of trading dates to keep the
    # panel balanced across models (required for fair cross-model comparison)
    common_dates = set.intersection(*[set(d) for d in date_sets])
    log(f"Aligning all tickers to {len(common_dates)} common trading dates.")
    df = df[df["Date"].isin(common_dates)]

# ------------------------------------------------------------------
# STEP 6: OHLC LOGICAL CONSISTENCY / OUTLIER CHECK
# ------------------------------------------------------------------
log("\nChecking OHLC logical consistency (High >= Low, High >= Open/Close, etc.)")
bad_rows = df[
    (df["High"] < df["Low"])
    | (df["High"] < df["Open"])
    | (df["High"] < df["Close"])
    | (df["Low"] > df["Open"])
    | (df["Low"] > df["Close"])
]
log(f"Rows violating OHLC logic: {len(bad_rows)}")
if len(bad_rows) > 0:
    df = df.drop(bad_rows.index)
    log(f"Removed. New shape: {df.shape}")

log("\nChecking for non-positive prices or negative volume...")
bad_price = df[(df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1) | (df["Volume"] < 0)]
log(f"Rows with non-positive price / negative volume: {len(bad_price)}")
if len(bad_price) > 0:
    df = df.drop(bad_price.index)

# Statistical outlier review: flag single-day returns beyond +/-40%
# (reviewed, NOT automatically removed - large moves are often genuine,
# e.g. NVDA post-earnings jumps, COVID crash days - removing them would
# bias the dataset. This is disclosed here for transparency per FR2.)
df = df.sort_values(["Ticker", "Date"])
df["daily_return"] = df.groupby("Ticker")["Close"].pct_change()
extreme_moves = df[df["daily_return"].abs() > 0.40]
log(f"\nDays with |return| > 40% (reviewed, retained as genuine market events): {len(extreme_moves)}")
if len(extreme_moves) > 0:
    log(extreme_moves[["Ticker", "Date", "Close", "daily_return"]].to_string())
df = df.drop(columns=["daily_return"])

# ------------------------------------------------------------------
# STEP 7: FORMATTING - add Sector column, sort, reset index
# ------------------------------------------------------------------
df["Sector"] = df["Ticker"].map(SECTOR_MAP)
df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

# ------------------------------------------------------------------
# STEP 8: SAVE CLEANED OUTPUTS
# ------------------------------------------------------------------
combined_path = os.path.join(OUTPUT_DIR, "cleaned_combined.csv")
df.to_csv(combined_path, index=False)
log(f"\nSaved combined cleaned dataset: {combined_path}  shape={df.shape}")

for ticker in TICKERS:
    sub = df[df["Ticker"] == ticker].copy()
    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_cleaned.csv")
    sub.to_csv(out_path, index=False)
    log(f"  Saved {ticker}: {out_path}  ({len(sub)} rows)")

# ------------------------------------------------------------------
# STEP 9: FINAL SUMMARY
# ------------------------------------------------------------------
log("\n" + "=" * 70)
log("FINAL CLEAN DATASET SUMMARY")
log("=" * 70)
log(f"Total rows: {len(df)}")
log(f"Tickers: {sorted(df['Ticker'].unique())}")
log(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
log(f"Sectors covered: {sorted(df['Sector'].unique())}")
log(f"Any remaining nulls: {df.isnull().sum().sum()}")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n\nCleaning report saved to {LOG_PATH}")
print("Data cleaning complete. Proceed to 02_feature_engineering.py")
