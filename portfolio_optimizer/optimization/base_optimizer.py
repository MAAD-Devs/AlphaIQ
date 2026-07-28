"""
Shared Abstract Base Class for portfolio optimization solvers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from ..core.data_models import OptimizationResult


class BasePortfolioOptimizer(ABC):
    """
    Abstract Base Class defining the contract for all portfolio solvers.
    """

    def __init__(self, risk_free_rate: float = 0.04, fee_drag_bps: float = 0.0):
        self.risk_free_rate = risk_free_rate
        self.fee_drag_bps = fee_drag_bps
        self.fee_drag_decimal = fee_drag_bps / 10000.0

    @abstractmethod
    def optimize(
        self,
        returns: pd.DataFrame,
        custom_cov: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        """
        Abstract method to solve portfolio weights.
        Returns OptimizationResult object.
        """
        pass

    def compute_summary_stats(
        self, weights: np.ndarray, mean_returns: pd.Series, cov_matrix: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculates annualized expected return, volatility, and Sharpe ratio given weights.
        """
        exp_ret = float(np.sum(mean_returns * weights) * 252 - self.fee_drag_decimal)
        port_var = float(weights.T @ cov_matrix.values @ weights * 252)
        port_vol = float(np.sqrt(max(1e-8, port_var)))
        sharpe = float((exp_ret - self.risk_free_rate) / port_vol) if port_vol > 0 else 0.0

        return {
            "expected_return": exp_ret,
            "volatility": port_vol,
            "sharpe_ratio": sharpe,
        }
