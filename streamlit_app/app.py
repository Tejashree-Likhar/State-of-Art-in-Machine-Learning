"""
State of the Art in Machine Learning — Stock Price Prediction
Streamlit dashboard.

Run with (from the project root):
    streamlit run streamlit_app/app.py

Reads results/metrics and results/predictions directly off disk (cached),
so there is no separate "build" step - re-run a model script, then use
Streamlit's rerun (press "R" or the ↻ in the top-right) to see new numbers.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================================================
# Paths & registry
# =============================================================
ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
PRED_DIR = ROOT / "results" / "predictions"

st.set_page_config(
    page_title="State of the Art in ML — Stock Prediction",
    page_icon="📈",
    layout="wide",
)

# Canonical model registry - edit here if you rename a model or add a new one.
# metricName / dirAccName must match the "Model" column text used in the
# corresponding results/metrics CSVs; predFilePrefix must match the prefix
# used in results/predictions/<prefix>_<TICKER>_predictions.csv.
MODEL_REGISTRY = [
    {"id": "Naive", "metricName": "Naive (Prev. Close)", "dirAccName": "Naive",
     "predFilePrefix": "Naive", "label": "Naive Baseline", "short": "Naive",
     "description": "Tomorrow's price = today's close. The floor every real model must beat.",
     "color": "#8B93A1"},
    {"id": "ARIMA", "metricName": "ARIMA", "dirAccName": "ARIMA",
     "predFilePrefix": "ARIMA", "label": "ARIMA", "short": "ARIMA",
     "description": "Classical autoregressive time-series model fit on price levels alone.",
     "color": "#4E88C7"},
    {"id": "LinearRegression", "metricName": "Linear Regression", "dirAccName": "LinearRegression",
     "predFilePrefix": "LinearRegression", "label": "Linear Regression", "short": "LinReg",
     "description": "OLS regression on engineered lag and technical-indicator features.",
     "color": "#B4923F"},
    {"id": "RandomForest", "metricName": "Random Forest", "dirAccName": "RandomForest",
     "predFilePrefix": "RandomForest", "label": "Random Forest", "short": "RF",
     "description": "Ensemble of decision trees, grid-searched, on the same engineered features.",
     "color": "#3E7C6F"},
    {"id": "XGBoost", "metricName": "XGBoost", "dirAccName": "XGBoost",
     "predFilePrefix": "XGBoost", "label": "XGBoost", "short": "XGB",
     "description": "Gradient-boosted trees on the same engineered feature set.",
     "color": "#A8503A"},
    {"id": "LSTM", "metricName": "LSTM", "dirAccName": "LSTM",
     "predFilePrefix": "LSTM", "label": "LSTM", "short": "LSTM",
     "description": "Recurrent neural network trained on windowed price sequences.",
     "color": "#8A5FB0"},
]
MODEL_BY_LABEL = {m["label"]: m for m in MODEL_REGISTRY}
LABEL_TO_SHORT = {m["label"]: m["short"] for m in MODEL_REGISTRY}
SECTOR_ORDER = ["Technology", "Healthcare", "Financials", "Energy", "Consumer Goods"]

DATASETS = {
    "primary": {
        "title": "S&P 500 OHLCV Historical Data",
        "link": "https://www.kaggle.com/datasets/jacksaleeby/s-and-p500-historical-data",
        "desc": "2.7M+ rows across 472 S&P 500 tickers, Jan 2000 – Feb 2026, originally "
                "sourced via yfinance from Yahoo Finance. Filtered down to the 10 target "
                "tickers and the Jan 2019 – Dec 2024 window. Verified clean: 0 duplicates, "
                "0 missing values, 0 OHLC logic violations, and already split-adjusted.",
        "stats": [("Rows (full)", "2.7M+"), ("Tickers (full)", "472"), ("Used window", "2019–2024")],
    },
    "secondary": {
        "title": "2019–2024 US Stock Market Data",
        "link": "https://www.kaggle.com/datasets/saketk511/2019-2024-us-stock-market-data",
        "desc": "Kept in data/raw/ for transparency, but excluded from modelling: it only "
                "contains 2 of the 10 required tickers (AAPL, NVDA), and is structured "
                "around commodities, crypto, and macro series rather than the 5-sector "
                "equity panel this dissertation requires.",
        "stats": [("Overlapping tickers", "2 / 10"), ("Focus", "Macro / crypto"), ("Verdict", "Excluded")],
    },
}

# =============================================================
# Data loading (cached — fast after first hit, always in sync with results/)
# =============================================================
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_predictions(prefix: str, ticker: str) -> pd.DataFrame:
    fp = PRED_DIR / f"{prefix}_{ticker}_predictions.csv"
    if not fp.exists():
        return pd.DataFrame(columns=["Date", "Actual", "Predicted"])
    df = pd.read_csv(fp, parse_dates=["Date"])
    return df


combined_df = load_csv(METRICS_DIR / "all_models_combined_metrics.csv")
overall_df = load_csv(METRICS_DIR / "overall_model_summary.csv")
best_df = load_csv(METRICS_DIR / "best_model_per_stock.csv")
diracc_df = load_csv(METRICS_DIR / "directional_accuracy_summary.csv")

if combined_df.empty:
    st.error(
        "Couldn't find results/metrics/all_models_combined_metrics.csv.\n\n"
        "Run this app from a project where the models have already been trained "
        "(see README §3), or check that streamlit_app/ sits inside the project "
        "root next to data/ and results/."
    )
    st.stop()

tickers_df = combined_df[["Ticker", "Sector"]].drop_duplicates().copy()
tickers_df["rank"] = tickers_df["Sector"].apply(lambda s: SECTOR_ORDER.index(s) if s in SECTOR_ORDER else 99)
tickers_df = tickers_df.sort_values(["rank", "Ticker"]).drop(columns="rank").reset_index(drop=True)
TICKERS = tickers_df["Ticker"].tolist()
BEST_MODEL_MAP = dict(zip(best_df["Ticker"], best_df["Model"])) if not best_df.empty else {}


def overall_value(model_id: str, metric: str):
    meta = next(m for m in MODEL_REGISTRY if m["id"] == model_id)
    if metric == "dirAcc":
        row = diracc_df[diracc_df["Model"] == meta["dirAccName"]]
        return float(row["Directional_Accuracy_Pct"].iloc[0]) if not row.empty else None
    row = overall_df[overall_df["Model"] == meta["metricName"]]
    if row.empty:
        return None
    col = {"rmse": "RMSE", "mae": "MAE", "mape": "MAPE", "r2": "R2"}[metric]
    return float(row[col].iloc[0])


def fmt(v, kind="num", decimals=3):
    if v is None or pd.isna(v):
        return "—"
    if kind == "pct":
        return f"{v:.{decimals}f}%"
    return f"{v:.{decimals}f}"


# =============================================================
# Custom styling (light touch — Streamlit's own theme does most of the work)
# =============================================================
st.markdown(
    """
    <style>
    .stApp { font-family: 'Source Sans Pro', sans-serif; }
    .sota-eyebrow {
        font-family: 'IBM Plex Mono', monospace; letter-spacing: .12em;
        text-transform: uppercase; font-size: 0.78rem; color: #B4923F; margin-bottom: 0.3rem;
    }
    .sota-finding {
        border-left: 3px solid #B4923F; padding: 0.6rem 1rem; background: #ffffffaa;
        border-radius: 4px; font-size: 1.05rem; margin: 0.8rem 0 1.2rem 0;
    }
    .dataset-card {
        border: 1px solid rgba(20,24,31,0.12); border-radius: 6px; padding: 1.1rem 1.3rem;
        background: #fff; height: 100%;
    }
    .dataset-card.used { border-left: 4px solid #3E7C6F; }
    .dataset-card.excluded { border-left: 4px solid #A8503A; }
    .dataset-tag {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: .06em; padding: 2px 8px; border-radius: 3px; display: inline-block; margin-bottom: 8px;
    }
    .dataset-card.used .dataset-tag { background: rgba(62,124,111,0.12); color: #3E7C6F; }
    .dataset-card.excluded .dataset-tag { background: rgba(168,80,58,0.1); color: #A8503A; }
    .ticker-chip {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; background: #E1E4DA;
        padding: 3px 9px; border-radius: 3px; margin: 2px 4px 2px 0; display: inline-block;
    }
    .best-badge {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; background: rgba(62,124,111,0.14);
        color: #3E7C6F; padding: 1px 6px; border-radius: 3px; margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================
# Header
# =============================================================
st.markdown('<div class="sota-eyebrow">MSc Data Science Dissertation · Implementation Dashboard</div>', unsafe_allow_html=True)
st.title("Six models. Ten stocks. One clear winner.")
st.caption(
    "A head-to-head comparison of classical, ensemble, and deep-learning approaches to "
    "next-day stock price prediction — trained and evaluated on six years of S&P 500 "
    "OHLCV data across five sectors."
)

best_id = max(MODEL_REGISTRY, key=lambda m: overall_value(m["id"], "r2") or -999)["id"]
best_meta = next(m for m in MODEL_REGISTRY if m["id"] == best_id)
wins = int((best_df["Model"] == best_meta["label"]).sum()) if not best_df.empty else 0
st.markdown(
    f'<div class="sota-finding"><b style="color:#B4923F">{best_meta["label"]}</b> beats every '
    f'other model on {wins} of {len(TICKERS)} stocks — the simplest model in the line-up '
    f"wins the whole comparison.</div>",
    unsafe_allow_html=True,
)

chip_cols = st.columns(len(MODEL_REGISTRY))
for col, m in zip(chip_cols, MODEL_REGISTRY):
    r2 = overall_value(m["id"], "r2")
    col.metric(m["short"], fmt(r2, decimals=3), help=f"Mean R² across all {len(TICKERS)} stocks")

st.divider()

# =============================================================
# Section navigation (segmented_control persists its choice in
# session_state via `key`, unlike st.tabs — which has no state of its
# own and can silently reset to the first tab on an unrelated rerun,
# e.g. when a widget inside another section changes).
# =============================================================
SECTIONS = ["📊 Model Comparison", "🗂️ Data Sources", "🔍 Model Deep Dive"]
active_section = st.segmented_control(
    "Section", SECTIONS, default=SECTIONS[0], required=True,
    key="active_section", label_visibility="collapsed",
)

# ---------------- Overview tab ----------------
if active_section == SECTIONS[0]:
    st.subheader("Mean error across all 10 stocks")
    st.caption("One bar per model. Switch metric to see the picture change — R² tells a very different story from RMSE.")

    metric_options = {
        "RMSE (lower is better)": ("rmse", False, "num"),
        "MAPE % (lower is better)": ("mape", False, "pct"),
        "R² (higher is better)": ("r2", True, "num"),
        "Directional accuracy % (higher is better)": ("dirAcc", True, "pct"),
    }
    metric_label = st.radio("Metric", list(metric_options.keys()), horizontal=True, label_visibility="collapsed")
    metric_key, higher_better, fmt_kind = metric_options[metric_label]

    bar_rows = []
    for m in MODEL_REGISTRY:
        val = overall_value(m["id"], metric_key)
        bar_rows.append({"label": m["label"], "short": m["short"], "value": val, "color": m["color"]})
    bar_df = pd.DataFrame(bar_rows).sort_values("value", ascending=not higher_better).reset_index(drop=True)
    best_bar = bar_df.iloc[0]
    plot_df = bar_df.iloc[::-1]  # reverse so the best bar renders at the top

    fig = go.Figure(
        go.Bar(
            x=plot_df["value"],
            y=plot_df["short"],
            orientation="h",
            marker_color=plot_df["color"],
            text=[fmt(v, fmt_kind) for v in plot_df["value"]],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis_title=metric_label,
        yaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono, monospace", size=13, color="#14181F"),
        xaxis=dict(gridcolor="rgba(20,24,31,0.08)", zerolinecolor="rgba(20,24,31,0.25)"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Sorted best → worst · best model here: **{best_bar['label']}**")

    st.subheader("Best model per stock")
    if not best_df.empty:
        st.dataframe(
            best_df.rename(columns={"RMSE": "RMSE", "MAPE": "MAPE", "R2": "R²"}),
            width="stretch",
            hide_index=True,
        )

# ---------------- Data sources tab ----------------
if active_section == SECTIONS[1]:
    st.subheader("Data sources")
    c1, c2 = st.columns(2)
    with c1:
        d = DATASETS["primary"]
        stats_html = "".join(f"<b>{v}</b><br><span style='color:#767D8C;font-size:.75rem'>{k}</span>&nbsp;&nbsp;&nbsp;" for k, v in d["stats"])
        st.markdown(
            f"""<div class="dataset-card used">
                <span class="dataset-tag">✓ used</span>
                <h4>{d['title']}</h4>
                <a href="{d['link']}" target="_blank">{d['link']}</a>
                <p style="color:#454C5A;font-size:.9rem;margin-top:10px;">{d['desc']}</p>
                <div style="margin-top:10px;font-family:'IBM Plex Mono',monospace;">{stats_html}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        d = DATASETS["secondary"]
        stats_html = "".join(f"<b>{v}</b><br><span style='color:#767D8C;font-size:.75rem'>{k}</span>&nbsp;&nbsp;&nbsp;" for k, v in d["stats"])
        st.markdown(
            f"""<div class="dataset-card excluded">
                <span class="dataset-tag">✕ excluded</span>
                <h4>{d['title']}</h4>
                <a href="{d['link']}" target="_blank">{d['link']}</a>
                <p style="color:#454C5A;font-size:.9rem;margin-top:10px;">{d['desc']}</p>
                <div style="margin-top:10px;font-family:'IBM Plex Mono',monospace;">{stats_html}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.subheader("Stock universe")
    st.caption(f"{len(TICKERS)} tickers across {tickers_df['Sector'].nunique()} sectors · 2 Jan 2019 – 31 Dec 2024")
    for sector in SECTOR_ORDER:
        syms = tickers_df.loc[tickers_df["Sector"] == sector, "Ticker"].tolist()
        if not syms:
            continue
        chips = "".join(f'<span class="ticker-chip">{s}</span>' for s in syms)
        st.markdown(f"**{sector}**&nbsp;&nbsp;{chips}", unsafe_allow_html=True)

# ---------------- Deep dive tab ----------------
if active_section == SECTIONS[2]:
    default_model_idx = [m["id"] for m in MODEL_REGISTRY].index(best_id)
    sel_col1, sel_col2 = st.columns([2, 1])
    with sel_col1:
        model_label = st.selectbox("Model", [m["label"] for m in MODEL_REGISTRY], index=default_model_idx, key="model_select")
    model = MODEL_BY_LABEL[model_label]
    st.caption(model["description"])

    stat_cols = st.columns(4)
    stat_cols[0].metric("Mean RMSE", fmt(overall_value(model["id"], "rmse"), decimals=2))
    stat_cols[1].metric("Mean MAPE", fmt(overall_value(model["id"], "mape"), "pct", 2))
    stat_cols[2].metric("Mean R²", fmt(overall_value(model["id"], "r2"), decimals=3))
    stat_cols[3].metric("Dir. Accuracy", fmt(overall_value(model["id"], "dirAcc"), "pct", 1))

    st.divider()

    # Per-stock table for this model (sortable natively — click any column header)
    model_rows = combined_df[combined_df["Model"] == model["metricName"]].copy()
    model_rows = model_rows.merge(tickers_df[["Ticker"]], on="Ticker", how="right")  # keep canonical order as fallback
    def _best_model_display(t):
        best_label = BEST_MODEL_MAP.get(t)
        if not best_label:
            return "—"
        short = LABEL_TO_SHORT.get(best_label, best_label)
        return f"★ {short}" if best_label == model["label"] else short

    model_rows["Best Model"] = model_rows["Ticker"].map(_best_model_display)
    model_rows = model_rows[["Ticker", "Sector", "RMSE", "MAE", "MAPE", "R2", "Best Model"]].rename(columns={"R2": "R²"})

    # The dropdown and the clickable table both drive the *same* chart, so they
    # share one canonical value in session_state instead of each keeping their
    # own — otherwise the dropdown can show one stock while the chart/table
    # show another (whichever control was used last "wins" invisibly).
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = TICKERS[0]
    if "_last_sel_rows" not in st.session_state:
        st.session_state._last_sel_rows = []

    stock_slot = sel_col2.empty()  # reserved now, filled in once the table below is read

    table_col, chart_col = st.columns([1, 1.3])
    with table_col:
        st.markdown(f"**{model_label} — per-stock results**")
        event = st.dataframe(
            model_rows.style.format({"RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE": "{:.3f}%", "R²": "{:.4f}"}),
            width="stretch",
            hide_index=True,
            height=390,
            on_select="rerun",
            selection_mode="single-row",
            key="metrics_table",
        )
        try:
            sel_rows = list(event.selection.rows) if hasattr(event, "selection") else list(event["selection"]["rows"])
        except Exception:
            sel_rows = []
        # Only react to a *new* click. Without this check, a selection left
        # over from before (e.g. after switching models) would keep firing
        # on every rerun and silently overrule the dropdown.
        if sel_rows and sel_rows != st.session_state._last_sel_rows:
            st.session_state.selected_ticker = model_rows.iloc[sel_rows[0]]["Ticker"]
        st.session_state._last_sel_rows = sel_rows
        st.caption("Tip: click a row to plot that stock, or use the dropdown above.")

    # Push the resolved ticker into the dropdown's own widget state *before*
    # creating it, so a table click visibly updates the dropdown too (setting
    # `index=` alone would be ignored once the widget already has a value).
    if st.session_state.get("stock_select") != st.session_state.selected_ticker:
        st.session_state["stock_select"] = st.session_state.selected_ticker

    def _sync_ticker_from_dropdown():
        st.session_state.selected_ticker = st.session_state["stock_select"]

    with stock_slot:
        st.selectbox("Stock", TICKERS, key="stock_select", on_change=_sync_ticker_from_dropdown)

    ticker = st.session_state.selected_ticker

    with chart_col:
        preds = load_predictions(model["predFilePrefix"], ticker)
        st.markdown(f"**{ticker} — Actual vs Predicted (test period)**")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=preds["Date"], y=preds["Actual"], name="Actual",
                                   line=dict(color="#14181F", width=1.8)))
        fig2.add_trace(go.Scatter(x=preds["Date"], y=preds["Predicted"], name="Predicted",
                                   line=dict(color=model["color"], width=1.8, dash="dot")))
        fig2.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="white",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            font=dict(family="IBM Plex Mono, monospace", size=11, color="#14181F"),
            xaxis=dict(gridcolor="rgba(20,24,31,0.05)"),
            yaxis=dict(title="Price ($)", gridcolor="rgba(20,24,31,0.08)"),
        )
        st.plotly_chart(fig2, width="stretch")

st.divider()
st.caption(
    "State of the Art in Machine Learning — Comparative Analysis of ML Models for Stock "
    "Price Prediction. MSc Data Science Dissertation, Heriot-Watt University — Tejashree Likhar. "
    "All figures read live from results/metrics and results/predictions."
)
