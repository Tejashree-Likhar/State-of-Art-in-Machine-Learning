"""
07_lstm_model.py
=================
LSTM deep learning model (FR13), following the AlQahtani et al. (2025)
architecture: 50 LSTM units, 0.2 dropout, dense output layer, Adam optimiser,
MSE loss, 60-timestep sliding window, Z-score normalised input (FR4).

IMPORTANT DESIGN NOTE ON SEQUENCE CONSTRUCTION:
To predict the FIRST test-set day, the model needs the 60 preceding days of
context - which fall inside the training period. This is standard practice
for time-series deep learning (not a data leak): the scaler itself is fit
ONLY on training data (NFR4), but once fitted, it is applied to transform
the full chronological series so that sliding windows can be built
continuously across the train/test boundary. Each window is then assigned
to the train or test split based on the date of the value it is predicting
(the window's target), never based on data it has not yet "seen" in time.

Runs on CPU by default; enable Google Colab GPU (see notebooks/04_lstm_colab.ipynb)
to meet the NFR1 training-time budget comfortably for all 10 stocks (NFR2).
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from utils import (
    load_ticker_data, chronological_split, scale_zscore, inverse_scale_target,
    evaluate_predictions, TICKERS, SECTOR_MAP, FEATURE_COLS, TARGET_COL,
    set_all_seeds, RANDOM_SEED, RESULTS_METRICS_DIR, RESULTS_PRED_DIR
)

set_all_seeds()

LOG_PATH = os.path.join(RESULTS_METRICS_DIR, "lstm_training_log.txt")
log_lines = []


def log(msg):
    print(msg)
    log_lines.append(str(msg))


WINDOW = 60
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.15  # last 15% of TRAINING windows used for early-stopping

INPUT_COLS = FEATURE_COLS + [TARGET_COL]  # Close is both an input feature and the target


def build_model(n_features):
    model = Sequential([
        Input(shape=(WINDOW, n_features)),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT_RATE),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")
    return model


log("=" * 70)
log("LSTM MODEL - TRAINING LOG")
log("=" * 70)
log(f"Architecture: LSTM({LSTM_UNITS}) -> Dropout({DROPOUT_RATE}) -> Dense(1)")
log(f"Window={WINDOW}, Optimizer=Adam(lr={LEARNING_RATE}), Loss=MSE")
log(f"Max epochs={MAX_EPOCHS}, EarlyStopping patience={EARLY_STOPPING_PATIENCE}")

# --- Checkpoint / resume support ---
METRICS_CSV = f"{RESULTS_METRICS_DIR}/LSTM_metrics.csv"
if os.path.exists(METRICS_CSV):
    all_metrics = pd.read_csv(METRICS_CSV).to_dict("records")
    done_tickers = set(r["Ticker"] for r in all_metrics)
else:
    all_metrics = []
    done_tickers = set()

if done_tickers:
    log(f"\nResuming: found existing results for {sorted(done_tickers)}. Skipping these.")

for ticker in TICKERS:
    if ticker in done_tickers:
        continue
    t0 = time.time()
    log(f"\n--- {ticker} ({SECTOR_MAP[ticker]}) ---")

    df = load_ticker_data(ticker)
    train_df, test_df = chronological_split(df)
    n_train = len(train_df)

    # Fit Z-score scaler on TRAIN ONLY (NFR4), then apply to full continuous series
    train_scaled, test_scaled, scaler = scale_zscore(train_df, test_df, INPUT_COLS)
    full_scaled = pd.concat([train_scaled, test_scaled], ignore_index=True)

    data = full_scaled[INPUT_COLS].values          # (n_total, n_features)
    target = full_scaled[TARGET_COL].values         # (n_total,)

    # Build sliding windows across the FULL continuous series
    X_all, y_all = [], []
    for i in range(WINDOW, len(data)):
        X_all.append(data[i - WINDOW:i])
        y_all.append(target[i])
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    # window i (0-indexed here) predicts original row index i + WINDOW
    target_row_idx = np.arange(WINDOW, len(data))

    train_mask = target_row_idx < n_train
    test_mask = target_row_idx >= n_train

    X_train_full, y_train_full = X_all[train_mask], y_all[train_mask]
    X_test, y_test_scaled_arr = X_all[test_mask], y_all[test_mask]

    # Carve chronological validation set from the END of the training windows
    n_val = int(len(X_train_full) * VALIDATION_FRACTION)
    X_fit, X_val = X_train_full[:-n_val], X_train_full[-n_val:]
    y_fit, y_val = y_train_full[:-n_val], y_train_full[-n_val:]

    log(f"Train windows={len(X_fit)}, Val windows={len(X_val)}, Test windows={len(X_test)}")

    tf.keras.backend.clear_session()
    set_all_seeds()
    model = build_model(n_features=len(INPUT_COLS))

    early_stop = EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True, verbose=0
    )

    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )
    n_epochs_trained = len(history.history["loss"])
    log(f"Stopped after {n_epochs_trained} epochs "
        f"(final val_loss={history.history['val_loss'][-1]:.6f})")

    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

    y_pred = inverse_scale_target(y_pred_scaled, scaler, INPUT_COLS, TARGET_COL)
    y_actual = test_df[TARGET_COL].values
    # sanity check: number of test predictions should equal test set length
    assert len(y_pred) == len(y_actual), (
        f"Mismatch: {len(y_pred)} predictions vs {len(y_actual)} actual test rows"
    )

    metrics = evaluate_predictions(y_actual, y_pred)
    elapsed = time.time() - t0
    log(f"RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"MAPE={metrics['MAPE']:.2f}%  R2={metrics['R2']:.4f}  "
        f"(trained in {elapsed:.1f}s)")

    all_metrics.append({
        "Model": "LSTM", "Ticker": ticker, "Sector": SECTOR_MAP[ticker],
        "Epochs_Trained": n_epochs_trained, **metrics, "Train_Time_Sec": elapsed
    })

    pred_df = pd.DataFrame({
        "Date": test_df["Date"].values,
        "Actual": y_actual,
        "Predicted": y_pred
    })
    pred_df.to_csv(f"{RESULTS_PRED_DIR}/LSTM_{ticker}_predictions.csv", index=False)

    # Save incrementally so progress is never lost
    pd.DataFrame(all_metrics).to_csv(METRICS_CSV, index=False)

metrics_df = pd.DataFrame(all_metrics)

log("\n" + "=" * 70)
log("LSTM SUMMARY (all tickers)")
log("=" * 70)
log(metrics_df[["Ticker", "Sector", "Epochs_Trained", "RMSE", "MAE", "MAPE", "R2", "Train_Time_Sec"]].to_string(index=False))
log(f"\nMean RMSE across all stocks: {metrics_df['RMSE'].mean():.4f}")
log(f"Mean MAPE across all stocks: {metrics_df['MAPE'].mean():.2f}%")

with open(LOG_PATH, "w") as f:
    f.write("\n".join(log_lines))

print(f"\n\nLSTM training log saved to {LOG_PATH}")
print("LSTM complete.")
