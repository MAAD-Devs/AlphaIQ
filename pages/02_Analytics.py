"""
Page 2: Quantitative Analytics & Tail-Risk Diagnostics.
"""

import os
import sys

# Ensure project root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_optimizer.analytics.factor_models import MarcenkoPasturDenoiser
from portfolio_optimizer.analytics.risk_metrics import (
    SharpeRatio,
    SortinoRatio,
    compute_all_risk_metrics,
)
from portfolio_optimizer.analytics.tail_risk import (
    CalmarRatio,
    ConditionalVaR,
    MaximumDrawdown,
    UlcerIndex,
    ValueAtRisk,
)
from utils.state_management import (
    fetch_and_cache_market_data,
    init_session_state,
    inject_custom_css,
    require_auth,
)

st.set_page_config(
    page_title="02 Analytics - Risk Diagnostics", page_icon="📊", layout="wide"
)
inject_custom_css()
init_session_state()
require_auth()

st.markdown(
    '<div class="gradient-header">02 Quantitative Risk & Diagnostics</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gradient-subtext">Deep-dive performance ratios, tail-risk metrics, correlation denoising, and drawdown analytics.</div>',
    unsafe_allow_html=True,
)

portfolio = st.session_state.portfolio
returns_df = st.session_state.returns_df
benchmark_series = st.session_state.benchmark_series
rf_rate = st.session_state.risk_free_rate

if returns_df.empty:
    st.warning("No market data available. Downloading latest market data...")
    fetch_and_cache_market_data()
    returns_df = st.session_state.returns_df

# Filter returns to portfolio tickers
portfolio_tickers = [col for col in portfolio.tickers if col in returns_df.columns]
if not portfolio_tickers:
    st.error("No valid ticker returns found in market data.")
    st.stop()

returns_df = returns_df[portfolio_tickers]
weights_arr = np.array([portfolio.weights.get(t, 0.0) for t in portfolio_tickers])
if np.sum(weights_arr) > 0:
    weights_arr /= np.sum(weights_arr)

# Compute portfolio daily return series
portfolio_daily_returns = returns_df.values @ weights_arr
port_series = pd.Series(
    portfolio_daily_returns, index=returns_df.index, name=portfolio.name
)

# --- Top  Performance Ratios ---
st.subheader("📌 Executive Risk & Return Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

sharpe = SharpeRatio(port_series, risk_free_rate=rf_rate)
sortino = SortinoRatio(port_series, risk_free_rate=rf_rate)
max_dd = MaximumDrawdown(port_series)
var_95 = ValueAtRisk(port_series, confidence_level=0.95)
cvar_95 = ConditionalVaR(port_series, confidence_level=0.95)

with col1:
    st.metric("Annualized Return", f"{port_series.mean() * 252 * 100:.2f}%")
with col2:
    st.metric("Annualized Volatility", f"{port_series.std() * np.sqrt(252) * 100:.2f}%")
with col3:
    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
with col4:
    st.metric("Sortino Ratio", f"{sortino:.2f}")
with col5:
    st.metric("Max Drawdown", f"{max_dd * 100:.2f}%")

st.markdown("---")

# --- Tab Layout ---
tab_breakdown, tab_tailrisk, tab_correlation, tab_denoise = st.tabs(
    [
        "📈 Asset-Level Breakdown",
        "⚠️ Tail Risk & Drawdowns",
        "🔥 Asset Correlation Matrix",
        "🧠 Random Matrix Denoising",
    ]
)

with tab_breakdown:
    st.subheader("Per-Asset Quantitative Diagnostics")

    metrics_list = []
    for ticker in portfolio_tickers:
        s = returns_df[ticker]
        stats = compute_all_risk_metrics(
            s, benchmark_returns=benchmark_series, risk_free_rate=rf_rate
        )
        stats["Ticker"] = ticker
        stats["Asset Class"] = next(
            (a.asset_class.value for a in portfolio.asset_values if a.ticker == ticker),
            "Unknown",
        )
        stats["Portfolio Weight (%)"] = portfolio.weights.get(ticker, 0.0) * 100
        metrics_list.append(stats)

    df_metrics = pd.DataFrame(metrics_list)

    # Reorder columns
    cols_order = [
        "Ticker",
        "Asset Class",
        "Portfolio Weight (%)",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "VaR 95%",
        "CVaR 95%",
    ]
    cols_present = [c for c in cols_order if c in df_metrics.columns]
    df_metrics = df_metrics[cols_present]

    st.dataframe(
        df_metrics.style.format(
            {
                "Portfolio Weight (%)": "{:.2f}%",
                "Annualized Return": "{:.2f}%",
                "Annualized Volatility": "{:.2f}%",
                "Sharpe Ratio": "{:.2f}",
                "Sortino Ratio": "{:.2f}",
                "Max Drawdown": "{:.2f}%",
                "VaR 95%": "{:.2f}%",
                "CVaR 95%": "{:.2f}%",
            }
        ),
        width="stretch",
    )

with tab_tailrisk:
    st.subheader("Tail Risk & Drawdown Profile")

    c_t1, c_t2, c_t3, c_t4 = st.columns(4)
    with c_t1:
        st.metric("Value at Risk (99%)", f"{ValueAtRisk(port_series, 0.99) * 100:.2f}%")
    with c_t2:
        st.metric(
            "Conditional VaR (99%)", f"{ConditionalVaR(port_series, 0.99) * 100:.2f}%"
        )
    with c_t3:
        st.metric("Calmar Ratio", f"{CalmarRatio(port_series):.2f}")
    with c_t4:
        st.metric("Ulcer Index", f"{UlcerIndex(port_series):.2f}")

    st.markdown("### Cumulative Drawdown Underwater Chart")
    cum_ret = (1 + port_series).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak

    try:
        import plotly.express as px

        fig_dd = px.area(
            drawdown,
            x=drawdown.index,
            y=drawdown.values,
            title="Portfolio Underwater Drawdown (%)",
            labels={"value": "Drawdown", "index": "Date"},
        )
        fig_dd.update_traces(fillcolor="rgba(239, 68, 68, 0.3)", line_color="#ef4444")
        fig_dd.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dd, width="stretch")
    except ImportError:
        st.area_chart(drawdown)

with tab_correlation:
    st.subheader("Asset Correlation Matrix")

    corr_matrix = returns_df.corr()
    try:
        import plotly.express as px

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Historical Return Correlation Heatmap",
        )
        fig_corr.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_corr, width="stretch")
    except ImportError:
        st.dataframe(
            corr_matrix.style.background_gradient(cmap="Blues"), width="stretch"
        )

with tab_denoise:
    st.subheader("Marcenko-Pastur Random Matrix Noise Denoising")
    st.write(
        "Filters out random sample noise from empirical covariance matrices using Random Matrix Theory (RMT)."
    )

    q_ratio = len(returns_df) / len(portfolio_tickers)
    denoiser = MarcenkoPasturDenoiser()
    denoised_cov = denoiser.denoise_covariance(returns_df.cov().values, q_ratio)

    df_denoised_cov = pd.DataFrame(
        denoised_cov, index=portfolio_tickers, columns=portfolio_tickers
    )
    st.write("**Denoised Covariance Matrix (Annualized)**:")
    st.dataframe((df_denoised_cov * 252).style.format("{:.4f}"), width="stretch")

st.markdown("---")
st.markdown(
    '<div class="fred-disclaimer">Notice: This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.</div>',
    unsafe_allow_html=True,
)
