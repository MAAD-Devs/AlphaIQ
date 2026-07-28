"""
Hierarchical Machine Learning portfolio optimizers: HRP (Hierarchical Risk Parity) & HERC.
"""

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

try:
    import riskfolio as rp
except ImportError:
    rp = None

from .base_optimizer import BasePortfolioOptimizer
from ..core.data_models import OptimizationResult, Portfolio, Asset


class HierarchicalRiskParityOptimizer(BasePortfolioOptimizer):
    """
    Hierarchical Risk Parity (HRP) & Hierarchical Equal Risk Contribution (HERC) solver.
    Uses Machine Learning linkage tree clustering on correlation distance matrices.
    """

    def __init__(
        self,
        method: str = "HRP",  # "HRP" or "HERC"
        linkage_method: str = "single",
        risk_free_rate: float = 0.04,
        fee_drag_bps: float = 0.0,
    ):
        super().__init__(risk_free_rate=risk_free_rate, fee_drag_bps=fee_drag_bps)
        self.method = method
        self.linkage_method = linkage_method

    def _hrp_recursive_bisection(self, cov: np.ndarray, sorted_indices: list) -> np.ndarray:
        """Applies recursive bisection algorithm on ordered assets tree."""
        weights = pd.Series(1.0, index=sorted_indices)
        clusters = [sorted_indices]

        while len(clusters) > 0:
            clusters = [
                cluster[j:k]
                for cluster in clusters
                for j, k in ((0, len(cluster) // 2), (len(cluster) // 2, len(cluster)))
                if len(cluster) > 1
            ]

            for i in range(0, len(clusters), 2):
                cluster_l = clusters[i]
                cluster_r = clusters[i + 1]

                cov_l = cov[np.ix_(cluster_l, cluster_l)]
                cov_r = cov[np.ix_(cluster_r, cluster_r)]

                # Cluster variance with inverse variance weighting
                w_l = 1.0 / np.diag(cov_l)
                w_l /= np.sum(w_l)
                var_l = float(w_l.T @ cov_l @ w_l)

                w_r = 1.0 / np.diag(cov_r)
                w_r /= np.sum(w_r)
                var_r = float(w_r.T @ cov_r @ w_r)

                alpha = 1.0 - (var_l / (var_l + var_r))
                weights[cluster_l] *= alpha
                weights[cluster_r] *= (1.0 - alpha)

        return weights.values

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
                port = rp.HCPortfolio(returns=returns)
                model = "HRP" if self.method == "HRP" else "HERC"
                w_df = port.optimization(model=model, rm="MV", rf=self.risk_free_rate, linkage=self.linkage_method)
                if w_df is not None:
                    w = w_df.values.flatten()
                    w_dict = dict(zip(tickers, w))
                    stats = self.compute_summary_stats(w, returns.mean(), returns.cov(), fee_drag=total_drag)
                    return OptimizationResult(
                        method=f"Hierarchical_{self.method}_Riskfolio",
                        weights=w_dict,
                        expected_return=stats["expected_return"],
                        volatility=stats["volatility"],
                        sharpe_ratio=stats["sharpe_ratio"],
                        status="Optimal",
                    )
            except Exception as e:
                print(f"Riskfolio HCPortfolio failed, using scipy linkage fallback: {e}")

        # Fallback linkage calculation
        cov_df = pd.DataFrame(custom_cov, index=tickers, columns=tickers) if custom_cov is not None else returns.cov()
        corr_df = returns.corr() if custom_cov is None else cov_df / np.outer(np.sqrt(np.diag(cov_df)), np.sqrt(np.diag(cov_df)))

        # Distance matrix d_ij = sqrt(0.5 * (1 - rho_ij))
        dist_matrix = np.sqrt(np.clip(0.5 * (1.0 - corr_df.values), 0, 1))
        np.fill_diagonal(dist_matrix, 0.0)

        condensed_dist = squareform(dist_matrix, checks=False)
        link = linkage(condensed_dist, method=self.linkage_method)
        ordered_idx = list(leaves_list(link))

        # Perform recursive bisection
        w_ordered = self._hrp_recursive_bisection(cov_df.values, ordered_idx)
        w = np.zeros(n)
        for idx, orig_i in enumerate(ordered_idx):
            w[orig_i] = w_ordered[idx]

        w_dict = dict(zip(tickers, w))
        stats = self.compute_summary_stats(w, returns.mean(), cov_df, fee_drag=total_drag)

        return OptimizationResult(
            method=f"Hierarchical_{self.method}",
            weights=w_dict,
            expected_return=stats["expected_return"],
            volatility=stats["volatility"],
            sharpe_ratio=stats["sharpe_ratio"],
            status="Optimal",
        )


if __name__ == "__main__":
    from ..core.data_models import AssetClass

    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    mock_returns = pd.DataFrame(
        np.random.normal(0.0005, 0.015, size=(252, 4)),
        index=dates,
        columns=["AAPL", "MSFT", "BND", "GLD"],
    )

    aapl = Asset("AAPL", AssetClass.EQUITY, "Apple Inc.")
    msft = Asset("MSFT", AssetClass.EQUITY, "Microsoft Corp.")
    bnd = Asset("BND", AssetClass.BOND, "Vanguard Total Bond Market ETF")
    gld = Asset("GLD", AssetClass.ETF, "SPDR Gold Shares")

    user_portfolio = Portfolio(
        name="Multi-Asset Portfolio",
        asset_values={aapl: 30000.0, msft: 30000.0, bnd: 20000.0, gld: 20000.0},
    )

    print("Testing HierarchicalRiskParityOptimizer...")

    # Test HRP
    hrp_opt = HierarchicalRiskParityOptimizer(method="HRP", risk_free_rate=0.04)
    res_hrp = hrp_opt.optimize(mock_returns, portfolio=user_portfolio)
    print("\n--- Hierarchical Risk Parity (HRP) ---")
    print(f"Method: {res_hrp.method}")
    print(f"Status: {res_hrp.status}")
    print(f"Expected Return: {res_hrp.expected_return:.4f}")
    print(f"Volatility: {res_hrp.volatility:.4f}")
    print(f"Sharpe Ratio: {res_hrp.sharpe_ratio:.4f}")
    print("Weights:", {k: round(v, 4) for k, v in res_hrp.weights.items()})

    # Test HERC
    herc_opt = HierarchicalRiskParityOptimizer(method="HERC", risk_free_rate=0.04)
    res_herc = herc_opt.optimize(mock_returns, portfolio=user_portfolio)
    print("\n--- Hierarchical Equal Risk Contribution (HERC) ---")
    print(f"Method: {res_herc.method}")
    print(f"Status: {res_herc.status}")
    print(f"Expected Return: {res_herc.expected_return:.4f}")
    print(f"Volatility: {res_herc.volatility:.4f}")
    print(f"Sharpe Ratio: {res_herc.sharpe_ratio:.4f}")
    print("Weights:", {k: round(v, 4) for k, v in res_herc.weights.items()})

