"""
Shared Abstract Base Class for portfolio optimization solvers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from ..core.data_models import OptimizationResult, Portfolio


class BasePortfolioOptimizer(ABC):
    """
    Abstract Base Class defining the contract for all portfolio solvers.
    """

    def __init__(self, risk_free_rate: float = 0.04, fee_drag_bps: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.fee_drag_bps = fee_drag_bps
        self.fee_drag_decimal = fee_drag_bps / 10000.0

    def compute_total_fee_drag(self, portfolio: Optional[Portfolio] = None) -> float:
        """
        Computes total fee drag decimal combining optimizer base fee drag, account drag, and asset annual drags.
        """
        total_drag = self.fee_drag_decimal
        if portfolio is not None:
            total_drag += portfolio.account_drag
            weights = portfolio.weights
            for asset in portfolio.asset_values:
                w = weights.get(asset.ticker, 0.0)
                total_drag += w * asset.annual_drag
        return total_drag

    def filter_returns_for_portfolio(
        self, returns: pd.DataFrame, portfolio: Optional[Portfolio] = None
    ) -> pd.DataFrame:
        """
        Filters returns DataFrame to match tickers present in the portfolio object if provided.
        """
        if portfolio is None:
            return returns

        portfolio_tickers = portfolio.tickers
        matching_cols = [col for col in portfolio_tickers if col in returns.columns]
        if matching_cols:
            return returns[matching_cols]
        return returns

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        custom_cov: Optional[np.ndarray] = None,
        portfolio: Optional[Portfolio] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Abstract method to solve portfolio weights.
        Returns OptimizationResult object.
        """
        pass

    def compute_summary_stats(
        self,
        weights: np.ndarray,
        mean_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        fee_drag: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Calculates annualized expected return, volatility, and Sharpe ratio given weights.
        """
        drag = self.fee_drag_decimal if fee_drag is None else fee_drag
        exp_ret = float(np.sum(mean_returns * weights) * 252 - drag)
        port_var = float(weights.T @ cov_matrix.values @ weights * 252)
        port_vol = float(np.sqrt(max(1e-8, port_var)))
        sharpe = float((exp_ret - self.risk_free_rate) / port_vol) if port_vol > 0 else 0.0

        return {
            "expected_return": exp_ret,
            "volatility": port_vol,
            "sharpe_ratio": sharpe,
        }
