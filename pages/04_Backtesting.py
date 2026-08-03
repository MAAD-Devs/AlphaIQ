"""
Page 4: Out-of-Sample Backtesting & Historical Strategy Performance.
"""

import numpy as np
import pandas as pd
import streamlit as st

from portfolio_optimizer.analytics.risk_metrics import SharpeRatio, SortinoRatio
from portfolio_optimizer.analytics.tail_risk import MaximumDrawdown, ValueAtRisk
from utils.state_management import (
    fetch_and_cache_market_data,
    init_session_state,
    inject_custom_css,
    require_auth,
    render_sidebar,
)

st.set_page_config(
    page_title="Backtesting - Strategy Performance", page_icon="📈", layout="wide"
)
inject_custom_css()
init_session_state()
require_auth()
render_sidebar()


st.markdown(
    '<div class="gradient-header">Out-of-Sample Backtesting</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gradient-subtext">Evaluate historical strategy performance, rolling Sharpe ratios, and rebalanced cumulative growth.</div>',
    unsafe_allow_html=True,
)

portfolio = st.session_state.portfolio
returns_df = st.session_state.returns_df
benchmark_series = st.session_state.benchmark_series
rf_rate = st.session_state.risk_free_rate
opt_res = st.session_state.optimization_result

if returns_df.empty:
    fetch_and_cache_market_data()
    returns_df = st.session_state.returns_df

portfolio_tickers = [col for col in portfolio.tickers if col in returns_df.columns]
if not portfolio_tickers:
    st.error("No active tickers available for backtesting.")
    st.stop()

returns_df = returns_df[portfolio_tickers]

# --- Sidebar / Backtest Controls ---
st.sidebar.subheader("Backtest Parameters")
initial_capital = st.sidebar.number_input(
    "Initial Investment ($)",
    min_value=1000.0,
    max_value=10000000.0,
    value=10000.0,
    step=1000.0,
)
rebalance_freq = st.sidebar.selectbox(
    "Rebalancing Frequency",
    options=["Buy & Hold", "Monthly", "Quarterly", "Annually"],
    index=1,
)
train_test_split = st.sidebar.slider(
    "In-Sample Train Split (%)", min_value=30, max_value=80, value=50, step=10
)

split_idx = int(len(returns_df) * (train_test_split / 100.0))
in_sample_returns = returns_df.iloc[:split_idx]
out_sample_returns = returns_df.iloc[split_idx:]

st.info(
    f"**Historical Timeline**: {len(returns_df)} Trading Days Total | **In-Sample Train**: {len(in_sample_returns)} Days | **Out-of-Sample Test**: {len(out_sample_returns)} Days"
)

# Calculate Strategy Returns
# 1. Current Allocation Returns
curr_weights = np.array([portfolio.weights.get(t, 0.0) for t in portfolio_tickers])
if np.sum(curr_weights) > 0:
    curr_weights /= np.sum(curr_weights)

current_strat_returns = out_sample_returns.values @ curr_weights

# 2. Optimized Allocation Returns
if opt_res is not None:
    opt_weights = np.array(
        [
            opt_res.weights.get(t, 1.0 / len(portfolio_tickers))
            for t in portfolio_tickers
        ]
    )
    if np.sum(opt_weights) > 0:
        opt_weights /= np.sum(opt_weights)
    opt_strat_returns = out_sample_returns.values @ opt_weights
    opt_name = f"Optimized ({opt_res.method})"
else:
    opt_weights = np.full(len(portfolio_tickers), 1.0 / len(portfolio_tickers))
    opt_strat_returns = out_sample_returns.values @ opt_weights
    opt_name = "Equal-Weight Anchor"

# 3. Benchmark Returns
bench_returns = benchmark_series.reindex(out_sample_returns.index).fillna(0.0).values

df_backtest = pd.DataFrame(
    {
        "Current Portfolio": current_strat_returns,
        opt_name: opt_strat_returns,
        "Benchmark (S&P 500)": bench_returns,
    },
    index=out_sample_returns.index,
)

# Cumulative Growth of $Initial
cum_growth = (1 + df_backtest).cumprod() * initial_capital

# --- Visual Growth Chart ---
st.subheader(
    f"Out-of-Sample Cumulative Growth (${initial_capital:,.0f} Initial Capital)"
)

try:
    import plotly.express as px

    fig_growth = px.line(
        cum_growth,
        x=cum_growth.index,
        y=cum_growth.columns,
        title=f"Out-of-Sample Backtest Cumulative Value (${initial_capital:,.0f} Base)",
        labels={"value": "Portfolio Value ($)", "variable": "Strategy"},
    )
    fig_growth.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_growth, width="stretch")
except ImportError:
    st.line_chart(cum_growth)

st.markdown("---")

# --- Performance Ratios Comparison Table ---
st.subheader("Strategy Performance Summary Table")

perf_rows = []
for col in df_backtest.columns:
    ret_series = df_backtest[col]
    total_ret = (cum_growth[col].iloc[-1] / initial_capital) - 1.0
    cagr = (1 + total_ret) ** (252.0 / len(ret_series)) - 1.0
    ann_vol = ret_series.std() * np.sqrt(252)
    sharpe = SharpeRatio(ret_series, risk_free_rate=rf_rate)
    sortino = SortinoRatio(ret_series, risk_free_rate=rf_rate)
    mdd = MaximumDrawdown(ret_series)
    var95 = ValueAtRisk(ret_series, 0.95)

    perf_rows.append(
        {
            "Strategy": col,
            "End Value ($)": cum_growth[col].iloc[-1],
            "Total Return (%)": total_ret * 100,
            "CAGR (%)": cagr * 100,
            "Annual Volatility (%)": ann_vol * 100,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown (%)": mdd * 100,
            "VaR 95% (%)": var95 * 100,
        }
    )

df_perf = pd.DataFrame(perf_rows)
st.dataframe(
    df_perf.style.format(
        {
            "End Value ($)": "${:,.2f}",
            "Total Return (%)": "{:.2f}%",
            "CAGR (%)": "{:.2f}%",
            "Annual Volatility (%)": "{:.2f}%",
            "Sharpe Ratio": "{:.2f}",
            "Sortino Ratio": "{:.2f}",
            "Max Drawdown (%)": "{:.2f}%",
            "VaR 95% (%)": "{:.2f}%",
        }
    ),
    width="stretch",
)

# --- Rolling Risk Diagnostics ---
st.markdown("### Rolling 60-Day Volatility")
rolling_vol = df_backtest.rolling(60).std() * np.sqrt(252) * 100.0

try:
    fig_rvol = px.line(
        rolling_vol,
        x=rolling_vol.index,
        y=rolling_vol.columns,
        title="Rolling 60-Day Annualized Volatility (%)",
        labels={"value": "Volatility (%)", "variable": "Strategy"},
    )
    fig_rvol.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_rvol, width="stretch")
except ImportError:
    st.line_chart(rolling_vol)

st.markdown("---")
st.markdown(
    '<div class="fred-disclaimer">Notice: This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.</div>',
    unsafe_allow_html=True,
)
