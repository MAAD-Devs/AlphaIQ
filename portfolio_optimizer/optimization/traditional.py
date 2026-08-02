"""
Traditional portfolio optimization solvers: Mean-Variance Optimization (MVO) & Vanilla Risk Parity.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    import riskfolio as rp
except ImportError:
    rp = None

from ..core.data_models import Asset, AssetClass, OptimizationResult, Portfolio
from .base_optimizer import BasePortfolioOptimizer


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
        portfolio: Optional[Portfolio] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        returns = self.filter_returns_for_portfolio(returns, portfolio)
        tickers = list(returns.columns)
        n = len(tickers)
        total_drag = self.compute_total_fee_drag(portfolio)

        if rp is not None and custom_cov is None:
            try:
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="hist")
                model = "Classic"
                obj = "Sharpe" if self.objective == "MaxSharpe" else "MinRisk"
                w_df = port.optimization(
                    model=model, rm="MV", obj=obj, rf=self.risk_free_rate, l=0
                )
                if w_df is not None:
                    w = w_df.values.flatten()
                    w_dict = dict(zip(tickers, w))
                    stats = self.compute_summary_stats(
                        w, returns.mean(), returns.cov(), fee_drag=total_drag
                    )
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
        cov = (
            pd.DataFrame(custom_cov, index=tickers, columns=tickers)
            if custom_cov is not None
            else returns.cov()
        )
        mean_ret = returns.mean()

        def min_vol_func(w):
            return np.sqrt(w.T @ cov.values @ w * 252)

        def max_sharpe_func(w):
            ret = np.sum(mean_ret * w) * 252 - total_drag
            vol = min_vol_func(w)
            return -(ret - self.risk_free_rate) / vol if vol > 0 else 0.0

        target_func = max_sharpe_func if self.objective == "MaxSharpe" else min_vol_func
        init_w = np.full(n, 1.0 / n)
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            target_func, init_w, bounds=bounds, constraints=constraints, method="SLSQP"
        )
        w = res.x if res.success else init_w
        w = w / np.sum(w)
        w_dict = dict(zip(tickers, w))

        stats = self.compute_summary_stats(w, mean_ret, cov, fee_drag=total_drag)
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
        portfolio: Optional[Portfolio] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        returns = self.filter_returns_for_portfolio(returns, portfolio)
        tickers = list(returns.columns)
        n = len(tickers)
        total_drag = self.compute_total_fee_drag(portfolio)

        if rp is not None and custom_cov is None:
            try:
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu="hist", method_cov="hist")
                w_df = port.rp_optimization(
                    model="Classic", rm="MV", rf=self.risk_free_rate
                )
                if w_df is not None:
                    w = w_df.values.flatten()
                    w_dict = dict(zip(tickers, w))
                    stats = self.compute_summary_stats(
                        w, returns.mean(), returns.cov(), fee_drag=total_drag
                    )
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

        cov = (
            pd.DataFrame(custom_cov, index=tickers, columns=tickers)
            if custom_cov is not None
            else returns.cov()
        )
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
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            risk_parity_objective,
            init_w,
            bounds=bounds,
            constraints=constraints,
            method="SLSQP",
        )
        w = res.x if res.success else init_w
        w = w / np.sum(w)
        w_dict = dict(zip(tickers, w))

        stats = self.compute_summary_stats(w, returns.mean(), cov, fee_drag=total_drag)
        return OptimizationResult(
            method="Vanilla_RiskParity",
            weights=w_dict,
            expected_return=stats["expected_return"],
            volatility=stats["volatility"],
            sharpe_ratio=stats["sharpe_ratio"],
            status="Optimal" if res.success else "Fallback_EqualWeight",
        )


if __name__ == "__main__":
    # Generate synthetic daily returns for 3 assets
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    mock_returns = pd.DataFrame(
        np.random.normal(0.0005, 0.015, size=(252, 3)),
        index=dates,
        columns=["AAPL", "MSFT", "GOOGL"],
    )

    # Define Portfolio domain object
    aapl = Asset("AAPL", AssetClass.EQUITY, "Apple Inc.", annual_drag=0.0)
    msft = Asset("MSFT", AssetClass.EQUITY, "Microsoft Corp.", annual_drag=0.0)
    googl = Asset("GOOGL", AssetClass.EQUITY, "Alphabet Inc.", annual_drag=0.0)
    user_portfolio = Portfolio(
        name="Tech Portfolio",
        asset_values={aapl: 50000.0, msft: 30000.0, googl: 20000.0},
        account_drag=0.001,
    )

    print("Testing MeanVarianceOptimizer with Portfolio data models...")

    # Test MaxSharpe
    mvo_max_sharpe = MeanVarianceOptimizer(objective="MaxSharpe", risk_free_rate=0.04)
    res_max_sharpe = mvo_max_sharpe.optimize(mock_returns, portfolio=user_portfolio)
    print("\n--- Max Sharpe Ratio ---")
    print(f"Method: {res_max_sharpe.method}")
    print(f"Status: {res_max_sharpe.status}")
    print(f"Expected Return: {res_max_sharpe.expected_return:.4f}")
    print(f"Volatility: {res_max_sharpe.volatility:.4f}")
    print(f"Sharpe Ratio: {res_max_sharpe.sharpe_ratio:.4f}")
    print("Weights:", {k: round(v, 4) for k, v in res_max_sharpe.weights.items()})

    # Test converting OptimizationResult to new Portfolio
    new_port = res_max_sharpe.to_portfolio(
        name="Rebalanced Portfolio", total_value=user_portfolio.total_value
    )
    print(f"\nNew Portfolio Total Value: ${new_port.total_value:,.2f}")
    print("New Portfolio Weights:", new_port.weights)

    # Test RiskParityOptimizer
    rp_opt = RiskParityOptimizer(risk_free_rate=0.04)
    res_rp = rp_opt.optimize(mock_returns, portfolio=user_portfolio)
    print("\n--- Risk Parity ---")
    print(f"Method: {res_rp.method}")
    print(f"Status: {res_rp.status}")
    print(f"Expected Return: {res_rp.expected_return:.4f}")
    print(f"Volatility: {res_rp.volatility:.4f}")
    print(f"Sharpe Ratio: {res_rp.sharpe_ratio:.4f}")
    print("Weights:", {k: round(v, 4) for k, v in res_rp.weights.items()})
