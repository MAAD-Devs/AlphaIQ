"""
Black-Litterman Bayesian portfolio optimization solver incorporating custom market views.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base_optimizer import BasePortfolioOptimizer
from ..core.data_models import OptimizationResult, Portfolio, Asset


class BlackLittermanOptimizer(BasePortfolioOptimizer):
    """
    Black-Litterman model combining market equilibrium prior returns with investor views.
    """

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
        risk_free_rate: float = 0.04,
        fee_drag_bps: float = 0.0,
    ):
        super().__init__(risk_free_rate=risk_free_rate, fee_drag_bps=fee_drag_bps)
        self.tau = tau
        self.risk_aversion = risk_aversion

    def optimize(
        self,
        returns: pd.DataFrame,
        custom_cov: Optional[np.ndarray] = None,
        market_weights: Optional[np.ndarray] = None,
        views_P: Optional[np.ndarray] = None,  # View pick matrix (k x n)
        views_Q: Optional[np.ndarray] = None,  # Expected view returns vector (k x 1)
        omega: Optional[np.ndarray] = None,    # View uncertainty matrix (k x k)
        portfolio: Optional[Portfolio] = None,
        **kwargs: Any,
    ) -> OptimizationResult:
        returns = self.filter_returns_for_portfolio(returns, portfolio)
        tickers = list(returns.columns)
        n = len(tickers)
        total_drag = self.compute_total_fee_drag(portfolio)

        cov_df = pd.DataFrame(custom_cov, index=tickers, columns=tickers) if custom_cov is not None else returns.cov()
        cov = cov_df.values

        # Market capitalization prior weights (default to portfolio weights or equal weights)
        if market_weights is None and portfolio is not None:
            port_weights = portfolio.weights
            w_mkt = np.array([port_weights.get(t, 1.0 / n) for t in tickers])
            if np.sum(w_mkt) > 0:
                w_mkt /= np.sum(w_mkt)
        else:
            w_mkt = market_weights if market_weights is not None else np.full(n, 1.0 / n)

        # Equilibrium prior returns: Pi = lambda * Sigma * w_mkt
        pi = self.risk_aversion * (cov @ w_mkt)

        # If no custom views provided, fallback to prior equilibrium
        if views_P is None or views_Q is None:
            mu_bl = pi
            cov_bl = cov
        else:
            k = len(views_Q)
            P = np.asarray(views_P)
            Q = np.asarray(views_Q).reshape(-1, 1)

            if omega is None:
                # Idzorek's heuristic / He & Litterman proportional uncertainty: Omega = diag(P * (tau * Sigma) * P^T)
                tau_sigma = self.tau * cov
                omega = np.diag(np.diag(P @ tau_sigma @ P.T))

            tau_sigma_inv = np.linalg.inv(self.tau * cov)
            omega_inv = np.linalg.inv(omega)

            # Posterior mean returns calculation
            M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
            mu_bl = M @ (tau_sigma_inv @ pi.reshape(-1, 1) + P.T @ omega_inv @ Q)
            mu_bl = mu_bl.flatten()

            # Posterior covariance
            cov_bl = cov + M

        # Solve for optimal weights given posterior mean and covariance
        def max_utility(w):
            ret = np.sum(mu_bl * w) * 252 - total_drag
            risk = 0.5 * self.risk_aversion * (w.T @ cov_bl @ w) * 252
            return -(ret - risk)

        init_w = np.full(n, 1.0 / n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(max_utility, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        w = res.x if res.success else init_w
        w = w / np.sum(w)
        w_dict = dict(zip(tickers, w))

        stats = self.compute_summary_stats(w, pd.Series(mu_bl / 252, index=tickers), cov_df, fee_drag=total_drag)
        return OptimizationResult(
            method="BlackLitterman",
            weights=w_dict,
            expected_return=stats["expected_return"],
            volatility=stats["volatility"],
            sharpe_ratio=stats["sharpe_ratio"],
            additional_metrics={
                "posterior_mean_returns": dict(zip(tickers, list(mu_bl * 252))),
                "equilibrium_prior_returns": dict(zip(tickers, list(pi * 252))),
            },
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

    print("Testing BlackLittermanOptimizer...")

    bl_opt = BlackLittermanOptimizer(tau=0.05, risk_aversion=2.5)
    # Define custom investor view: AAPL will outperform MSFT by 5% annually (relative view)
    P = np.array([[1, -1, 0]])
    Q = np.array([0.05])

    res = bl_opt.optimize(
        returns=mock_returns,
        portfolio=user_portfolio,
        views_P=P,
        views_Q=Q,
    )

    print(f"\nMethod: {res.method}")
    print(f"Status: {res.status}")
    print(f"Expected Return: {res.expected_return:.4f}")
    print(f"Volatility: {res.volatility:.4f}")
    print(f"Sharpe Ratio: {res.sharpe_ratio:.4f}")
    print("Weights:", {k: round(v, 4) for k, v in res.weights.items()})
    print("Posterior Mean Returns:", {k: round(v, 4) for k, v in res.additional_metrics["posterior_mean_returns"].items()})

