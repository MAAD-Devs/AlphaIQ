"""
Kelly Criterion logarithmic utility optimizer for maximizing long-term CAGR.
"""

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base_optimizer import BasePortfolioOptimizer
from ..core.data_models import OptimizationResult, Portfolio, Asset


class KellyCriterionOptimizer(BasePortfolioOptimizer):
    """
    Logarithmic utility portfolio solver for maximizing expected compound growth rate (CAGR).
    Supports fractional Kelly scaling (e.g., 0.5 for Half-Kelly).
    """

    def __init__(
        self,
        fraction: float = 0.5,  # 0.5 = Half-Kelly to reduce volatility drawdown
        risk_free_rate: float = 0.04,
        fee_drag_bps: float = 0.0,
    ):
        super().__init__(risk_free_rate=risk_free_rate, fee_drag_bps=fee_drag_bps)
        self.fraction = fraction

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
        r_matrix = returns.values

        # Log utility objective: max E[ln(1 + R_p)]
        def obj_func(w):
            port_returns = r_matrix @ w
            # Prevent log(<= 0)
            valid_returns = np.maximum(1 + port_returns, 1e-6)
            mean_log_ret = np.mean(np.log(valid_returns))
            return -mean_log_ret

        init_w = np.full(n, 1.0 / n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(obj_func, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        w_full = res.x if res.success else init_w
        w_full = w_full / np.sum(w_full)

        # Apply fractional Kelly scaling relative to equal weight anchor or cash
        w_scaled = self.fraction * w_full + (1.0 - self.fraction) * (np.full(n, 1.0 / n))
        w_scaled = w_scaled / np.sum(w_scaled)

        w_dict = dict(zip(tickers, w_scaled))
        stats = self.compute_summary_stats(w_scaled, returns.mean(), returns.cov(), fee_drag=total_drag)

        return OptimizationResult(
            method=f"KellyCriterion_frac_{self.fraction}",
            weights=w_dict,
            expected_return=stats["expected_return"],
            volatility=stats["volatility"],
            sharpe_ratio=stats["sharpe_ratio"],
            additional_metrics={"full_kelly_weights": dict(zip(tickers, list(w_full)))},
            status="Optimal" if res.success else "Fallback_EqualWeight",
        )


if __name__ == "__main__":
    from ..core.data_models import AssetClass

    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    mock_returns = pd.DataFrame(
        np.random.normal(0.0008, 0.018, size=(252, 3)),
        index=dates,
        columns=["AAPL", "TSLA", "NVDA"],
    )

    aapl = Asset("AAPL", AssetClass.EQUITY, "Apple Inc.")
    tsla = Asset("TSLA", AssetClass.EQUITY, "Tesla Inc.")
    nvda = Asset("NVDA", AssetClass.EQUITY, "NVIDIA Corp.")

    user_portfolio = Portfolio(
        name="Growth Tech Portfolio",
        asset_values={aapl: 40000.0, tsla: 30000.0, nvda: 30000.0},
    )

    print("Testing KellyCriterionOptimizer...")

    kelly_opt = KellyCriterionOptimizer(fraction=0.5, risk_free_rate=0.04)
    res_kelly = kelly_opt.optimize(mock_returns, portfolio=user_portfolio)

    print(f"\nMethod: {res_kelly.method}")
    print(f"Status: {res_kelly.status}")
    print(f"Expected Return: {res_kelly.expected_return:.4f}")
    print(f"Volatility: {res_kelly.volatility:.4f}")
    print(f"Sharpe Ratio: {res_kelly.sharpe_ratio:.4f}")
    print("Half-Kelly Weights:", {k: round(v, 4) for k, v in res_kelly.weights.items()})
    print("Full-Kelly Weights:", {k: round(v, 4) for k, v in res_kelly.additional_metrics["full_kelly_weights"].items()})

