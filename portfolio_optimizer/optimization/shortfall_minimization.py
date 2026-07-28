"""
Shortfall minimization optimizer targeting CVaR (Expected Shortfall) and CDaR (Conditional Drawdown at Risk).
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
from ..core.data_models import OptimizationResult, Portfolio, Asset
from ..analytics.tail_risk import ConditionalVaR, MaximumDrawdown


class ShortfallMinimizationOptimizer(BasePortfolioOptimizer):
    """
    Solves for portfolio weights that minimize CVaR (Conditional Value at Risk) or CDaR (Conditional Drawdown at Risk).
    """

    def __init__(
        self,
        risk_measure: str = "CVaR",  # "CVaR" or "CDaR"
        alpha: float = 0.95,
        risk_free_rate: float = 0.04,
        fee_drag_bps: float = 0.0,
    ):
        super().__init__(risk_free_rate=risk_free_rate, fee_drag_bps=fee_drag_bps)
        self.risk_measure = risk_measure
        self.alpha = alpha

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
                rm_choice = "CVaR" if self.risk_measure == "CVaR" else "CDaR"
                w_df = port.optimization(model="Classic", rm=rm_choice, obj="MinRisk", rf=self.risk_free_rate, alpha=1 - self.alpha)
                if w_df is not None:
                    w = w_df.values.flatten()
                    w_dict = dict(zip(tickers, w))
                    stats = self.compute_summary_stats(w, returns.mean(), returns.cov(), fee_drag=total_drag)
                    return OptimizationResult(
                        method=f"ShortfallMinimization_{self.risk_measure}_Riskfolio",
                        weights=w_dict,
                        expected_return=stats["expected_return"],
                        volatility=stats["volatility"],
                        sharpe_ratio=stats["sharpe_ratio"],
                        status="Optimal",
                    )
            except Exception as e:
                print(f"Riskfolio Shortfall Optimization failed, using scipy fallback: {e}")

        r_matrix = returns.values

        # Objective function for empirical CVaR / CDaR minimization
        def obj_func(w):
            port_returns = r_matrix @ w
            if self.risk_measure == "CVaR":
                return ConditionalVaR(port_returns, confidence_level=self.alpha)
            else:  # CDaR / Drawdown
                return MaximumDrawdown(port_returns, is_returns=True)

        init_w = np.full(n, 1.0 / n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(obj_func, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        w = res.x if res.success else init_w
        w = w / np.sum(w)
        w_dict = dict(zip(tickers, w))

        stats = self.compute_summary_stats(w, returns.mean(), returns.cov(), fee_drag=total_drag)
        return OptimizationResult(
            method=f"ShortfallMinimization_{self.risk_measure}",
            weights=w_dict,
            expected_return=stats["expected_return"],
            volatility=stats["volatility"],
            sharpe_ratio=stats["sharpe_ratio"],
            additional_metrics={f"minimized_{self.risk_measure}": float(obj_func(w))},
            status="Optimal" if res.success else "Fallback_EqualWeight",
        )


if __name__ == "__main__":
    from ..core.data_models import AssetClass

    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    mock_returns = pd.DataFrame(
        np.random.normal(0.0005, 0.015, size=(252, 3)),
        index=dates,
        columns=["AAPL", "MSFT", "GOOGL"],
    )

    aapl = Asset("AAPL", AssetClass.EQUITY, "Apple Inc.")
    msft = Asset("MSFT", AssetClass.EQUITY, "Microsoft Corp.")
    googl = Asset("GOOGL", AssetClass.EQUITY, "Alphabet Inc.")

    user_portfolio = Portfolio(
        name="Tech Portfolio",
        asset_values={aapl: 50000.0, msft: 30000.0, googl: 20000.0},
    )

    print("Testing ShortfallMinimizationOptimizer...")

    # Test CVaR
    cvar_opt = ShortfallMinimizationOptimizer(risk_measure="CVaR", alpha=0.95, risk_free_rate=0.04)
    res_cvar = cvar_opt.optimize(mock_returns, portfolio=user_portfolio)
    print("\n--- CVaR Minimization ---")
    print(f"Method: {res_cvar.method}")
    print(f"Status: {res_cvar.status}")
    print(f"Expected Return: {res_cvar.expected_return:.4f}")
    print(f"Volatility: {res_cvar.volatility:.4f}")
    print(f"Sharpe Ratio: {res_cvar.sharpe_ratio:.4f}")
    print("Weights:", {k: round(v, 4) for k, v in res_cvar.weights.items()})

    # Test CDaR
    cdar_opt = ShortfallMinimizationOptimizer(risk_measure="CDaR", alpha=0.95, risk_free_rate=0.04)
    res_cdar = cdar_opt.optimize(mock_returns, portfolio=user_portfolio)
    print("\n--- CDaR Minimization ---")
    print(f"Method: {res_cdar.method}")
    print(f"Status: {res_cdar.status}")
    print(f"Expected Return: {res_cdar.expected_return:.4f}")
    print(f"Volatility: {res_cdar.volatility:.4f}")
    print(f"Sharpe Ratio: {res_cdar.sharpe_ratio:.4f}")
    print("Weights:", {k: round(v, 4) for k, v in res_cdar.weights.items()})

