"""
Traditional portfolio optimization solvers: Mean-Variance Optimization (MVO) & Vanilla Risk Parity.
"""

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import riskfolio as rp
except ImportError:
    rp = None

from .base_optimizer import BasePortfolioOptimizer
from ..core.data_models import OptimizationResult


class MeanVarianceOptimizer(BasePortfolioOptimizer):
    """
    Mean-Variance Optimization (MVO) solver for Max Sharpe Ratio or Minimum Volatility.
    Supports Riskfolio-Lib backend with scipy fallback.
    """

    def __init__(
        self,
        objective: str = "MaxSharpe",  # "MaxSharpe" or "MinVol"
        risk_free_rate: float = 0.04,
        fee_drag_bps: float = 0.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ):
        super().__init__(risk_free_rate=risk_free_rate, fee_drag_bps=fee_drag_bps)
        self.objective = objective
        self.min_weight = min_weight
        self.max_weight = max_weight

    def optimize(
        self,
        returns: pd.DataFrame,
        custom_cov: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        tickers = list(returns.columns)
        n = len(tickers)

        if rp is not None and custom_cov is None:
            try:
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="hist")
                model = "Classic"
                obj = "Sharpe" if self.objective == "MaxSharpe" else "MinRisk"
                w_df = port.optimization(model=model, rm="MV", obj=obj, rf=self.risk_free_rate, l=0)
                if w_df is not None:
                    w = w_df.values.flatten()
                    w_dict = dict(zip(tickers, w))
                    stats = self.compute_summary_stats(w, returns.mean(), returns.cov())
                    return OptimizationResult(
                        method=f"MVO_{self.objective}_Riskfolio",
                        weights=w_dict,
                        expected_return=stats["expected_return"],
                        volatility=stats["volatility"],
                        sharpe_ratio=stats["sharpe_ratio"],
                        status="Optimal",
                    )
            except Exception as e:
                print(f"Riskfolio solver failed, using scipy fallback: {e}")

        # SciPy fallback
        cov = pd.DataFrame(custom_cov, index=tickers, columns=tickers) if custom_cov is not None else returns.cov()
        mean_ret = returns.mean()

        def min_vol_func(w):
            return np.sqrt(w.T @ cov.values @ w * 252)

        def max_sharpe_func(w):
            ret = np.sum(mean_ret * w) * 252 - self.fee_drag_decimal
            vol = min_vol_func(w)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 0.0

        target_func = max_sharpe_func if self.objective == "MaxSharpe" else min_vol_func
        init_w = np.full(n, 1.0 / n)
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(target_func, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        w = res.x if res.success else init_w
        w = w / np.sum(w)
        w_dict = dict(zip(tickers, w))

        stats = self.compute_summary_stats(w, mean_ret, cov)
        return OptimizationResult(
            method=f"MVO_{self.objective}",
            weights=w_dict,
            expected_return=stats["expected_return"],
            volatility=stats["volatility"],
            sharpe_ratio=stats["sharpe_ratio"],
            status="Optimal" if res.success else "Fallback_EqualWeight",
        )


class RiskParityOptimizer(BasePortfolioOptimizer):
    """
    Equal Risk Contribution (Vanilla Risk Parity) solver.
    """

    def optimize(
        self,
        returns: pd.DataFrame,
        custom_cov: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        tickers = list(returns.columns)
        n = len(tickers)

        if rp is not None and custom_cov is None:
            try:
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="hist")
                w_df = port.rp_optimization(model="Classic", rm="MV", rf=self.risk_free_rate)
                if w_df is not None:
                    w = w_df.values.flatten()
                    w_dict = dict(zip(tickers, w))
                    stats = self.compute_summary_stats(w, returns.mean(), returns.cov())
                    return OptimizationResult(
                        method="RiskParity_Riskfolio",
                        weights=w_dict,
                        expected_return=stats["expected_return"],
                        volatility=stats["volatility"],
                        sharpe_ratio=stats["sharpe_ratio"],
                        status="Optimal",
                    )
            except Exception as e:
                print(f"Riskfolio RiskParity failed, using scipy fallback: {e}")

        cov = pd.DataFrame(custom_cov, index=tickers, columns=tickers) if custom_cov is not None else returns.cov()
        cov_vals = cov.values

        # Equal Risk Contribution objective: min sum( (w_i - (sigma^2 w)_i / (w' Sigma w))^2 )
        def risk_parity_objective(w):
            port_vol = np.sqrt(w.T @ cov_vals @ w)
            marginal_risk = (cov_vals @ w) / port_vol
            risk_contrib = w * marginal_risk
            target_risk = port_vol / n
            return np.sum((risk_contrib - target_risk) ** 2)

        init_w = np.full(n, 1.0 / n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(risk_parity_objective, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        w = res.x if res.success else init_w
        w = w / np.sum(w)
        w_dict = dict(zip(tickers, w))

        stats = self.compute_summary_stats(w, returns.mean(), cov)
        return OptimizationResult(
            method="Vanilla_RiskParity",
            weights=w_dict,
            expected_return=stats["expected_return"],
            volatility=stats["volatility"],
            sharpe_ratio=stats["sharpe_ratio"],
            status="Optimal" if res.success else "Fallback_EqualWeight",
        )
