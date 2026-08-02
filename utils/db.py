import logging
from typing import Optional

import streamlit as st
from supabase import Client, create_client

from portfolio_optimizer.core.data_models import Asset, AssetClass, Portfolio

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_client() -> Client:
    return create_client(st.secrets["supabase"]["url"], st.secrets["supabase"]["key"])


def get_or_create_user(email: str) -> None:
    _get_client().table("users").upsert({"email": email}, on_conflict="email").execute()


def load_active_portfolio(email: str) -> Optional[Portfolio]:
    client = _get_client()

    port_resp = (
        client.table("portfolios")
        .select("*")
        .eq("user_email", email)
        .eq("is_active", True)
        .execute()
    )
    if not port_resp.data:
        return None

    port_row = port_resp.data[0]
    assets_resp = (
        client.table("portfolio_assets")
        .select("*")
        .eq("portfolio_id", port_row["id"])
        .execute()
    )

    asset_values: dict[Asset, float] = {}
    for row in assets_resp.data:
        try:
            ac = AssetClass(row["asset_class"])
        except ValueError:
            ac = AssetClass.EQUITY
        asset = Asset(
            ticker=row["ticker"],
            asset_class=ac,
            name=row.get("display_name") or "",
            annual_drag=float(row.get("annual_drag", 0.0)),
        )
        asset_values[asset] = float(row["dollar_value"])

    return Portfolio(
        name=port_row["name"],
        asset_values=asset_values,
        account_drag=float(port_row.get("account_drag", 0.0)),
    )


def save_portfolio(email: str, portfolio: Portfolio) -> None:
    client = _get_client()

    existing = (
        client.table("portfolios")
        .select("id")
        .eq("user_email", email)
        .eq("is_active", True)
        .execute()
    )

    if existing.data:
        portfolio_id = existing.data[0]["id"]
        client.table("portfolios").update(
            {"name": portfolio.name, "account_drag": portfolio.account_drag}
        ).eq("id", portfolio_id).execute()
    else:
        result = (
            client.table("portfolios")
            .insert(
                {
                    "user_email": email,
                    "name": portfolio.name,
                    "account_drag": portfolio.account_drag,
                    "is_active": True,
                }
            )
            .execute()
        )
        portfolio_id = result.data[0]["id"]

    client.table("portfolio_assets").delete().eq("portfolio_id", portfolio_id).execute()

    rows = [
        {
            "portfolio_id": portfolio_id,
            "ticker": asset.ticker,
            "asset_class": asset.asset_class.value,
            "display_name": asset.name,
            "annual_drag": asset.annual_drag,
            "dollar_value": dollar_value,
        }
        for asset, dollar_value in portfolio.asset_values.items()
    ]
    if rows:
        client.table("portfolio_assets").insert(rows).execute()
