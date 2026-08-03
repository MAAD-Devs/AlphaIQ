"""
Executive Overview & Main App Launcher for Investment Portfolio Optimizer.
"""

import os
import sys

# Ensure project root directory is in sys.path to resolve 'portfolio_optimizer' package imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

# Import domain models and state management
from utils.state_management import (
    PRESET_TEMPLATES,
    fetch_and_cache_market_data,
    init_session_state,
    inject_custom_css,
    load_portfolio_template,
    persist_portfolio,
    render_sidebar,
)

# Configure Streamlit page
st.set_page_config(
    page_title="Portfolio Optimizer & Risk Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS & initialize state
inject_custom_css()
init_session_state()

# --- Auth Gate ---
if hasattr(st.user, "is_logged_in") and not st.user.is_logged_in:
    st.markdown(
        '<div class="gradient-header">Quantitative Portfolio Optimizer</div>',
        unsafe_allow_html=True,
    )
    st.markdown("Please sign in to continue.")
    if st.button("Sign in with Google"):
        st.login("google")
    st.stop()

# Ensure market data is loaded
if st.session_state.returns_df.empty:
    fetch_and_cache_market_data()

# --- Sidebar Controls ---
render_sidebar()


# --- Header Section ---
st.markdown(
    '<div class="gradient-header">Quantitative Portfolio Optimizer</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gradient-subtext">Institutional-grade risk hedging, Bayesian allocation, and out-of-sample backtesting dashboard.</div>',
    unsafe_allow_html=True,
)

portfolio = st.session_state.portfolio

# --- Top Key Performance Indicators (KPIs) ---
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Total Portfolio Value",
        value=f"${portfolio.total_value:,.2f}",
        delta=f"{len(portfolio.tickers)} Assets",
    )

with col2:
    st.metric(
        label="Account Fee Drag",
        value=f"{portfolio.account_drag * 10000:.1f} bps",
        delta=f"{(portfolio.account_drag * 100):.2f}% / yr",
    )

with col3:
    st.metric(
        label="Risk-Free Benchmark Rate",
        value=f"{st.session_state.risk_free_rate * 100:.2f}%",
        delta="Treasuries Anchor",
    )

with col4:
    st.metric(
        label="Data Lookback Horizon",
        value=st.session_state.lookback_period.upper(),
        delta=f"{len(st.session_state.returns_df)} Trading Days",
    )

with col5:
    st.metric(
        label="Primary Benchmark",
        value="S&P 500 (^GSPC)",
        delta="Equilibrium Prior",
    )

st.markdown("---")

# --- Main Dashboard Grid ---
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("Portfolio Asset Allocation Breakdown")

    weights = portfolio.weights
    df_weights = pd.DataFrame(
        [
            {
                "Ticker": asset.ticker,
                "Name": asset.name or asset.ticker,
                "Asset Class": asset.asset_class.value,
                "Value ($)": val,
                "Weight (%)": weights.get(asset.ticker, 0.0) * 100,
                "Annual Expense Drag (%)": asset.annual_drag * 100,
            }
            for asset, val in portfolio.asset_values.items()
        ]
    )

    if not df_weights.empty:
        # Display Interactive Chart
        try:
            import plotly.express as px

            fig = px.pie(
                df_weights,
                names="Ticker",
                values="Value ($)",
                hole=0.4,
                color="Asset Class",
                title=f"Allocation by Ticker - Total Value: ${portfolio.total_value:,.2f}",
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, width="stretch")
        except ImportError:
            st.bar_chart(df_weights.set_index("Ticker")["Weight (%)"])

        st.dataframe(
            df_weights.style.format(
                {
                    "Value ($)": "${:,.2f}",
                    "Weight (%)": "{:.2f}%",
                    "Annual Expense Drag (%)": "{:.3f}%",
                }
            ),
            width="stretch",
        )

with right_col:
    st.subheader("⚡ Core Modules Navigation")
    st.markdown(
        """
        Use the multipage navigation in the sidebar or jump directly to key workflows:

        * **Data Ingestion & Entry**: Update portfolio holdings, edit ticker quantities, set fee drag parameters, and review live price data quality.
        * **Quantitative Analytics**: Inspect Sharpe/Sortino ratios, tail-risk metrics (VaR/CVaR/Max Drawdown), factor models, and fixed-income duration metrics.
        * **Portfolio Optimization**: Execute Mean-Variance, Risk Parity, Black-Litterman, HRP, Kelly Criterion, or CVaR/CDaR minimization with linear constraints.
        * **Out-of-Sample Backtesting**: Test strategy rebalancing frequencies and benchmark relative performance across historical regimes.
        """
    )

    st.info(
        "💡 **Quick Tip**: Navigate to **01 Data Ingestion** to modify your holdings or add new tickers to your custom portfolio!"
    )

st.markdown("---")

# MANDATORY FRED Attribution
st.markdown(
    '<div class="fred-disclaimer">Notice: This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.</div>',
    unsafe_allow_html=True,
)
