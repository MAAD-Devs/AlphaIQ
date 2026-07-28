"""
Black-Litterman Bayesian portfolio optimization solver incorporating custom market views.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base_optimizer import BasePortfolioOptimizer
from ..core.data_models import OptimizationResult


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
        **kwargs: Any,
    ) -> OptimizationResult:
        tickers = list(returns.columns)
        n = len(tickers)

        cov_df = pd.DataFrame(custom_cov, index=tickers, columns=tickers) if custom_cov is not None else returns.cov()
        cov = cov_df.values

        # Market capitalization prior weights (default to equal weights if missing)
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
            ret = np.sum(mu_bl * w) * 252 - self.fee_drag_decimal
            risk = 0.5 * self.risk_aversion * (w.T @ cov_bl @ w) * 252
            return -(ret - risk)

        init_w = np.full(n, 1.0 / n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        res = minimize(max_utility, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
        w = res.x if res.success else init_w
        w = w / np.sum(w)
        w_dict = dict(zip(tickers, w))

        stats = self.compute_summary_stats(w, pd.Series(mu_bl / 252, index=tickers), cov_df)
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
