"""
Constraint builder and manager: Linear constraints, sector caps, Beta caps, and fee drag adjustments.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd


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

    def apply_fee_drag(self, expected_returns: pd.Series, turnover: Optional[float] = None) -> pd.Series:
        """
        Adjusts expected returns series by subtracting fee drag basis points and turnover cost.
        """
        fee_drag_decimal = self.fee_drag_bps / 10000.0
        adjusted_returns = expected_returns - fee_drag_decimal
        if turnover is not None:
            adjusted_returns -= (turnover * fee_drag_decimal)
        return adjusted_returns
