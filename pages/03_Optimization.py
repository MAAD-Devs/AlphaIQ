"""
Page 3: Portfolio Optimization Engine & Rebalancing Interface.
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

from portfolio_optimizer.optimization import (
    BlackLittermanOptimizer,
    HierarchicalRiskParityOptimizer,
    KellyCriterionOptimizer,
    MeanVarianceOptimizer,
    RiskParityOptimizer,
    ShortfallMinimizationOptimizer,
)
from utils.state_management import (
    fetch_and_cache_market_data,
    init_session_state,
    inject_custom_css,
    require_auth,
)

st.set_page_config(
    page_title="03 Optimization - Portfolio Solvers", page_icon="🎯", layout="wide"
)
inject_custom_css()
init_session_state()
require_auth()

st.markdown(
    '<div class="gradient-header">03 Portfolio Optimization Engine</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gradient-subtext">Execute institutional allocation solvers: MVO, Risk Parity, Black-Litterman, HRP, Kelly Criterion, and CVaR Minimization.</div>',
    unsafe_allow_html=True,
)

portfolio = st.session_state.portfolio
returns_df = st.session_state.returns_df
rf_rate = st.session_state.risk_free_rate

if returns_df.empty:
    fetch_and_cache_market_data()
    returns_df = st.session_state.returns_df

portfolio_tickers = [col for col in portfolio.tickers if col in returns_df.columns]
if not portfolio_tickers:
    st.error("No active tickers available in market data.")
    st.stop()

returns_df = returns_df[portfolio_tickers]

# --- Sidebar / Control Panel ---
col_controls, col_results = st.columns([1, 2])

with col_controls:
    st.subheader("⚙️ Solver Configuration")

    solver_type = st.selectbox(
        "Select Optimization Model",
        options=[
            "Mean-Variance (Max Sharpe)",
            "Mean-Variance (Min Volatility)",
            "Risk Parity (Equal Risk Contribution)",
            "Black-Litterman (Bayesian Views)",
            "Hierarchical Risk Parity (HRP)",
            "Hierarchical Equal Risk (HERC)",
            "Kelly Criterion (CAGR Growth)",
            "Shortfall Minimization (CVaR 95%)",
            "Shortfall Minimization (CDaR Drawdown)",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("🔒 Constraints & Parameters")

    min_weight = st.slider(
        "Min Asset Weight",
        min_value=0.0,
        max_value=0.3,
        value=0.0,
        step=0.05,
        format="%.2f",
    )
    max_weight = st.slider(
        "Max Asset Weight",
        min_value=0.2,
        max_value=1.0,
        value=1.0,
        step=0.05,
        format="%.2f",
    )

    # Specific Model Options
    views_P = None
    views_Q = None
    kelly_frac = 0.5

    if "Black-Litterman" in solver_type:
        st.info("💡 Black-Litterman View Matrix Settings")
        v_asset1 = st.selectbox("Asset View Ticker", options=portfolio_tickers, index=0)
        v_asset2 = st.selectbox(
            "Outperformed Ticker",
            options=portfolio_tickers,
            index=min(1, len(portfolio_tickers) - 1),
        )
        outperform_pct = st.number_input(
            "Expected Outperformance (%/yr)",
            min_value=-0.20,
            max_value=0.30,
            value=0.05,
            step=0.01,
        )

        idx1 = portfolio_tickers.index(v_asset1)
        idx2 = portfolio_tickers.index(v_asset2)
        k_vec = np.zeros(len(portfolio_tickers))
        k_vec[idx1] = 1.0
        k_vec[idx2] = -1.0
        views_P = np.array([k_vec])
        views_Q = np.array([outperform_pct])

    elif "Kelly" in solver_type:
        kelly_frac = st.slider(
            "Kelly Fraction (0.5 = Half-Kelly)",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
        )

    btn_run = st.button("🚀 Run Optimization", width="stretch")

# Handle Optimization Execution
if btn_run or st.session_state.optimization_result is None:
    with st.spinner("Solving optimal portfolio weights..."):
        if solver_type == "Mean-Variance (Max Sharpe)":
            solver = MeanVarianceOptimizer(
                objective="MaxSharpe",
                min_weight=min_weight,
                max_weight=max_weight,
                risk_free_rate=rf_rate,
            )
            res = solver.optimize(returns_df, portfolio=portfolio)

        elif solver_type == "Mean-Variance (Min Volatility)":
            solver = MeanVarianceOptimizer(
                objective="MinVol",
                min_weight=min_weight,
                max_weight=max_weight,
                risk_free_rate=rf_rate,
            )
            res = solver.optimize(returns_df, portfolio=portfolio)

        elif solver_type == "Risk Parity (Equal Risk Contribution)":
            solver = RiskParityOptimizer(risk_free_rate=rf_rate)
            res = solver.optimize(returns_df, portfolio=portfolio)

        elif solver_type == "Black-Litterman (Bayesian Views)":
            solver = BlackLittermanOptimizer(risk_free_rate=rf_rate)
            res = solver.optimize(
                returns_df, portfolio=portfolio, views_P=views_P, views_Q=views_Q
            )

        elif solver_type == "Hierarchical Risk Parity (HRP)":
            solver = HierarchicalRiskParityOptimizer(
                method="HRP", risk_free_rate=rf_rate
            )
            res = solver.optimize(returns_df, portfolio=portfolio)

        elif solver_type == "Hierarchical Equal Risk (HERC)":
            solver = HierarchicalRiskParityOptimizer(
                method="HERC", risk_free_rate=rf_rate
            )
            res = solver.optimize(returns_df, portfolio=portfolio)

        elif solver_type == "Kelly Criterion (CAGR Growth)":
            solver = KellyCriterionOptimizer(
                fraction=kelly_frac, risk_free_rate=rf_rate
            )
            res = solver.optimize(returns_df, portfolio=portfolio)

        elif solver_type == "Shortfall Minimization (CVaR 95%)":
            solver = ShortfallMinimizationOptimizer(
                risk_measure="CVaR", alpha=0.95, risk_free_rate=rf_rate
            )
            res = solver.optimize(returns_df, portfolio=portfolio)

        else:  # CDaR
            solver = ShortfallMinimizationOptimizer(
                risk_measure="CDaR", alpha=0.95, risk_free_rate=rf_rate
            )
            res = solver.optimize(returns_df, portfolio=portfolio)

        st.session_state.optimization_result = res

res = st.session_state.optimization_result

# --- Results Output Section ---
with col_results:
    st.subheader(f"📊 Optimization Output: {res.method}")

    # Top Ratios Row
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Expected Return", f"{res.expected_return * 100:.2f}%")
    with r2:
        st.metric("Expected Volatility", f"{res.volatility * 100:.2f}%")
    with r3:
        st.metric("Sharpe Ratio", f"{res.sharpe_ratio:.2f}")
    with r4:
        st.metric("Solver Status", res.status)

    st.markdown("---")

    # Comparison Table (Current vs Target Weights)
    curr_weights = portfolio.weights
    total_val = portfolio.total_value

    rows = []
    for ticker in portfolio_tickers:
        w_curr = curr_weights.get(ticker, 0.0)
        w_opt = res.weights.get(ticker, 0.0)
        val_curr = w_curr * total_val
        val_opt = w_opt * total_val
        diff_val = val_opt - val_curr

        rows.append(
            {
                "Ticker": ticker,
                "Current Weight (%)": w_curr * 100,
                "Optimized Weight (%)": w_opt * 100,
                "Current Value ($)": val_curr,
                "Optimized Value ($)": val_opt,
                "Rebalance Action ($)": diff_val,
            }
        )

    df_comp = pd.DataFrame(rows)

    # Bar chart comparison
    try:
        import plotly.graph_objects as go

        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                x=df_comp["Ticker"],
                y=df_comp["Current Weight (%)"],
                name="Current Weight (%)",
                marker_color="#6366f1",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                x=df_comp["Ticker"],
                y=df_comp["Optimized Weight (%)"],
                name="Optimized Weight (%)",
                marker_color="#10b981",
            )
        )
        fig_bar.update_layout(
            barmode="group",
            title="Current vs Optimized Allocation Weights",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, width="stretch")
    except ImportError:
        st.bar_chart(
            df_comp.set_index("Ticker")[["Current Weight (%)", "Optimized Weight (%)"]]
        )

    st.subheader("📋 Detailed Rebalancing Trade Schedule")
    st.dataframe(
        df_comp.style.format(
            {
                "Current Weight (%)": "{:.2f}%",
                "Optimized Weight (%)": "{:.2f}%",
                "Current Value ($)": "${:,.2f}",
                "Optimized Value ($)": "${:,.2f}",
                "Rebalance Action ($)": "${:+,.2f}",
            }
        ),
        width="stretch",
    )

    if st.button(
        "✅ Apply Rebalanced Weights to Active Portfolio",
        type="primary",
        width="stretch",
    ):
        new_portfolio = res.to_portfolio(
            name=f"{portfolio.name} (Optimized - {res.method})",
            total_value=total_val,
            asset_map={a.ticker: a for a in portfolio.asset_values},
            account_drag=portfolio.account_drag,
        )
        st.session_state.portfolio = new_portfolio
        fetch_and_cache_market_data()
        st.success("Active portfolio rebalanced and updated successfully!")
        st.rerun()

st.markdown("---")
st.markdown(
    '<div class="fred-disclaimer">Notice: This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.</div>',
    unsafe_allow_html=True,
)
