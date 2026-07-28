"""
Constraint builder and manager: Linear constraints, sector caps, Beta caps, and fee drag adjustments.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from ..core.data_models import Portfolio, Asset, AssetClass


class PortfolioConstraints:
    """
    Constructs constraint matrices and parameter dicts for portfolio solvers.
    Handles asset bounds, sector caps, Beta limits relative to benchmark, and fee drag cost adjustments.
    """

    def __init__(
        self,
        asset_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        sector_caps: Optional[Dict[str, float]] = None,
        max_beta: Optional[float] = None,
        fee_drag_bps: float = 10.0,
    ):
        self.asset_bounds = asset_bounds or {}
        self.sector_caps = sector_caps or {}
        self.max_beta = max_beta
        self.fee_drag_bps = fee_drag_bps

    def build_scipy_bounds(self, tickers: List[str], default_min: float = 0.0, default_max: float = 1.0) -> Tuple[Tuple[float, float], ...]:
        """Generates SciPy solver bounds tuple for each ticker."""
        bounds = []
        for ticker in tickers:
            b = self.asset_bounds.get(ticker, (default_min, default_max))
            bounds.append(b)
        return tuple(bounds)

    def build_sector_constraints(
        self, tickers: List[str], asset_sectors: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """
        Generates linear inequality constraints for sector allocation caps.
        Sum(w_i for i in sector) <= cap
        """
        constraints = []
        for sector, cap in self.sector_caps.items():
            mask = np.array([1.0 if asset_sectors.get(t) == sector else 0.0 for t in tickers])
            constraints.append({
                'type': 'ineq',
                'fun': lambda w, m=mask, c=cap: c - np.sum(w * m)
            })
        return constraints

    def build_sector_constraints_from_portfolio(
        self, portfolio: Portfolio
    ) -> List[Dict[str, Any]]:
        """
        Generates sector constraints directly from a Portfolio object's asset classes.
        """
        tickers = portfolio.tickers
        asset_sectors = {asset.ticker: asset.asset_class.value for asset in portfolio.asset_values}
        return self.build_sector_constraints(tickers, asset_sectors)

    def build_beta_constraint(
        self, tickers: List[str], asset_betas: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """
        Generates linear inequality constraint for overall portfolio Beta cap vs benchmark.
        Sum(w_i * Beta_i) <= max_beta
        """
        if self.max_beta is None:
            return None

        beta_vec = np.array([asset_betas.get(t, 1.0) for t in tickers])
        return {
            'type': 'ineq',
            'fun': lambda w, b=beta_vec, cap=self.max_beta: cap - np.sum(w * b)
        }

    def apply_fee_drag(self, expected_returns: pd.Series, turnover: Optional[float] = None, portfolio: Optional[Portfolio] = None) -> pd.Series:
        """
        Adjusts expected returns series by subtracting fee drag basis points, asset annual drag, and turnover cost.
        """
        fee_drag_decimal = self.fee_drag_bps / 10000.0
        if portfolio is not None:
            fee_drag_decimal += portfolio.account_drag
            weights = portfolio.weights
            for asset in portfolio.asset_values:
                fee_drag_decimal += weights.get(asset.ticker, 0.0) * asset.annual_drag

        adjusted_returns = expected_returns - fee_drag_decimal
        if turnover is not None:
            adjusted_returns -= (turnover * (self.fee_drag_bps / 10000.0))
        return adjusted_returns


if __name__ == "__main__":
    print("Testing PortfolioConstraints...")

    aapl = Asset("AAPL", AssetClass.EQUITY, "Apple Inc.", annual_drag=0.0)
    bnd = Asset("BND", AssetClass.BOND, "Vanguard Total Bond Market ETF", annual_drag=0.0003)

    user_portfolio = Portfolio(
        name="Test Portfolio",
        asset_values={aapl: 60000.0, bnd: 40000.0},
        account_drag=0.001,
    )

    constraints = PortfolioConstraints(
        asset_bounds={"AAPL": (0.1, 0.7), "BND": (0.2, 0.8)},
        sector_caps={"Equity": 0.7, "Bond": 0.5},
        max_beta=1.1,
        fee_drag_bps=15.0,
    )

    tickers = user_portfolio.tickers
    bounds = constraints.build_scipy_bounds(tickers)
    print("SciPy Bounds:", bounds)

    sector_constraints = constraints.build_sector_constraints_from_portfolio(user_portfolio)
    print("Sector Constraints Count:", len(sector_constraints))

    raw_returns = pd.Series([0.12, 0.04], index=tickers)
    adjusted_returns = constraints.apply_fee_drag(raw_returns, portfolio=user_portfolio)
    print("Raw Returns:\n", raw_returns.to_dict())
    print("Adjusted Returns:\n", adjusted_returns.to_dict())

