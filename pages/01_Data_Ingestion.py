"""
Page 1: Portfolio Data Entry & Data Ingestion Pipeline.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Ensure project root directory is in sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pandas as pd
import streamlit as st

from portfolio_optimizer.core.data_models import Asset, AssetClass, Portfolio
from utils.state_management import (
    PRESET_TEMPLATES,
    fetch_and_cache_market_data,
    init_session_state,
    inject_custom_css,
    load_portfolio_template,
    persist_portfolio,
    require_auth,
)

st.set_page_config(
    page_title="01 Data Ingestion - Portfolio Entry", page_icon="📥", layout="wide"
)
inject_custom_css()
init_session_state()
require_auth()

st.markdown(
    '<div class="gradient-header">01 Portfolio Entry & Data Ingestion</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="gradient-subtext">Manage user portfolio holdings, configure fee drags, and inspect market data quality.</div>',
    unsafe_allow_html=True,
)

portfolio = st.session_state.portfolio

# --- Tab Layout ---
tab_entry, tab_inspection, tab_presets = st.tabs(
    ["Portfolio Data Entry", "Market Data Inspection", "Presets & Templates"]
)

with tab_entry:
    st.subheader("Current Portfolio Holdings")

    # Display portfolio metadata inputs
    col_name, col_drag = st.columns(2)
    with col_name:
        new_port_name = st.text_input("Portfolio Name", value=portfolio.name)
    with col_drag:
        new_account_drag_bps = st.number_input(
            "Account Fee Drag (Basis Points)",
            min_value=0.0,
            max_value=500.0,
            value=float(portfolio.account_drag * 10000),
            step=5.0,
            help="Annual fee drag incurred at the account level (e.g., AUM fee or SDBA wrapper fee).",
        )

    # Build editable table representation
    asset_rows = []
    for asset, val in portfolio.asset_values.items():
        asset_rows.append(
            {
                "Ticker": asset.ticker,
                "Asset Class": asset.asset_class.value,
                "Name": asset.name,
                "Dollar Value ($)": float(val),
                "Annual Drag (bps)": float(asset.annual_drag * 10000),
            }
        )

    df_editable = pd.DataFrame(asset_rows)

    edited_df = st.data_editor(
        df_editable,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker (e.g. AAPL)", required=True),
            "Asset Class": st.column_config.SelectboxColumn(
                "Asset Class",
                options=[ac.value for ac in AssetClass],
                required=True,
            ),
            "Name": st.column_config.TextColumn("Asset Description"),
            "Dollar Value ($)": st.column_config.NumberColumn(
                "Dollar Value ($)", min_value=0.0, format="$%.2f"
            ),
            "Annual Drag (bps)": st.column_config.NumberColumn(
                "Expense Drag (bps)", min_value=0.0, format="%.1f bps"
            ),
        },
        key="portfolio_table_editor",
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("💾 Save Portfolio Changes"):
            new_asset_values = {}
            for idx, row in edited_df.iterrows():
                t = str(row["Ticker"]).strip().upper()
                if not t:
                    continue
                ac_str = row["Asset Class"]
                try:
                    ac = AssetClass(ac_str)
                except ValueError:
                    ac = AssetClass.EQUITY

                name = str(row.get("Name", ""))
                val = float(row.get("Dollar Value ($)", 0.0))
                drag_bps = float(row.get("Annual Drag (bps)", 0.0))

                asset_obj = Asset(
                    ticker=t,
                    asset_class=ac,
                    name=name,
                    annual_drag=drag_bps / 10000.0,
                )
                new_asset_values[asset_obj] = val

            st.session_state.portfolio = Portfolio(
                name=new_port_name,
                asset_values=new_asset_values,
                account_drag=new_account_drag_bps / 10000.0,
            )
            persist_portfolio(st.session_state.portfolio)
            fetch_and_cache_market_data()
            st.success("Portfolio saved!")
            st.rerun()

with tab_inspection:
    st.subheader("Market Data Quality & Diagnostic Report")

    if st.session_state.returns_df.empty:
        st.info("Click 'Refresh Market Data' to fetch historical price data.")
    else:
        prices_df = st.session_state.prices_df
        returns_df = st.session_state.returns_df

        st.write(f"**Loaded Tickers**: {list(returns_df.columns)}")
        st.write(
            f"**Date Range**: {returns_df.index.min().strftime('%Y-%m-%d')} to {returns_df.index.max().strftime('%Y-%m-%d')} ({len(returns_df)} observations)"
        )

        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            st.metric("Missing Data Points", int(prices_df.isna().sum().sum()))
        with col_q2:
            st.metric("Daily Return Std Dev", f"{returns_df.std().mean() * 100:.2f}%")
        with col_q3:
            st.metric(
                "Avg Annualized Return", f"{returns_df.mean().mean() * 252 * 100:.2f}%"
            )

        st.markdown("### Normalized Historical Price Performance (Base 100)")
        norm_prices = (prices_df / prices_df.iloc[0]) * 100.0

        try:
            import plotly.express as px

            fig_prices = px.line(
                norm_prices,
                x=norm_prices.index,
                y=norm_prices.columns,
                title="Historical Rebased Price Trends (Base = 100)",
                labels={"value": "Rebased Index", "variable": "Ticker"},
            )
            fig_prices.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                hovermode="x unified",
            )
            st.plotly_chart(fig_prices, width="stretch")
        except ImportError:
            st.line_chart(norm_prices)

with tab_presets:
    st.subheader("Quick Portfolio Presets")
    st.write("Select a pre-configured institutional asset allocation template:")

    for key, info in PRESET_TEMPLATES.items():
        st.markdown(f"#### {key}")
        st.write(
            f"**Name**: {info['name']} | **Account Drag**: {info['account_drag'] * 10000:.0f} bps"
        )

        p_df = pd.DataFrame(
            [
                {
                    "Ticker": a.ticker,
                    "Class": a.asset_class.value,
                    "Name": a.name,
                    "Value ($)": v,
                }
                for a, v in zip(info["assets"], info["values"])
            ]
        )
        st.dataframe(p_df, width="stretch")

        if st.button(f"Load {key} Template", key=f"btn_load_{key}"):
            st.session_state.portfolio = load_portfolio_template(key)
            persist_portfolio(st.session_state.portfolio)
            fetch_and_cache_market_data()
            st.success(f"Successfully loaded '{key}' portfolio!")
            st.rerun()

st.markdown("---")
st.markdown(
    '<div class="fred-disclaimer">Notice: This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis.</div>',
    unsafe_allow_html=True,
)
