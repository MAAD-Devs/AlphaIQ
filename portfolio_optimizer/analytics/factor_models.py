"""
Fama-French 5-Factor regression engine and Marcenko-Pastur random matrix covariance denoising.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import statsmodels.api as sm


class FamaFrench5Factor:
    """
    Fits Fama-French 5-Factor model (Mkt-RF, SMB, HML, RMW, CMA) using OLS regression.
    """

    def __init__(self, factor_data: Optional[pd.DataFrame] = None):
        self.factor_data = factor_data

    def _generate_synthetic_factors(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generates synthetic factor series if official factor data is not supplied."""
        np.random.seed(42)
        n = len(dates)
        df = pd.DataFrame(
            {
                "Mkt-RF": np.random.normal(0.0004, 0.01, n),
                "SMB": np.random.normal(0.0001, 0.005, n),
                "HML": np.random.normal(0.0001, 0.006, n),
                "RMW": np.random.normal(0.0002, 0.004, n),
                "CMA": np.random.normal(0.0001, 0.004, n),
                "RF": np.full(n, 0.04 / 252),
            },
            index=dates,
        )
        return df

    def fit(
        self, asset_returns: pd.Series, factors: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Regresses asset returns against Fama-French 5 factors.
        """
        if factors is None:
            if self.factor_data is not None:
                factors = self.factor_data
            else:
                factors = self._generate_synthetic_factors(asset_returns.index)

        # Align indices
        common_idx = asset_returns.index.intersection(factors.index)
        y = asset_returns.loc[common_idx] - factors.loc[common_idx, "RF"]
        X = factors.loc[common_idx, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]]
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        betas = model.params.to_dict()
        pvalues = model.pvalues.to_dict()

        return {
            "alpha": float(betas.pop("const", 0.0)),
            "betas": {k: float(v) for k, v in betas.items()},
            "p_values": {k: float(v) for k, v in pvalues.items()},
            "r_squared": float(model.rsquared),
            "adj_r_squared": float(model.rsquared_adj),
            "f_statistic": float(model.fvalue) if model.fvalue is not None else 0.0,
        }


class MarcenkoPasturDenoiser:
    """
    Applies Marcenko-Pastur theorem to denoise sample covariance matrices by removing noise eigenvalues.
    """

    @staticmethod
    def fit_marcenko_pastur(
        var: float, q: float, pts: int = 1000
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Calculates theoretical Marcenko-Pastur distribution limits (lambda_min, lambda_max).
        q = T / N (number of observations / number of assets)
        """
        e_min = var * (1 - np.sqrt(1.0 / q)) ** 2
        e_max = var * (1 + np.sqrt(1.0 / q)) ** 2
        e_val = np.linspace(e_min, e_max, pts)
        pdf = q / (2 * np.pi * var * e_val) * np.sqrt((e_max - e_val) * (e_val - e_min))
        pdf = np.nan_to_num(pdf)
        return e_min, e_max, e_val, pdf

    def denoise_covariance(
        self, cov: np.ndarray, q: float, shrinkage: bool = True
    ) -> np.ndarray:
        """
        Denoises sample covariance matrix using Spectral Shrinkage based on Marcenko-Pastur boundary.
        """
        # Convert covariance to correlation matrix
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        corr = np.nan_to_num(corr)

        # Eigenvalue decomposition
        e_val, e_vec = np.linalg.eigh(corr)
        idx = e_val.argsort()[::-1]
        e_val = e_val[idx]
        e_vec = e_vec[:, idx]

        # Estimate Marcenko-Pastur upper bound
        e_min, e_max, _, _ = self.fit_marcenko_pastur(var=1.0, q=q)

        # Replace noise eigenvalues with average of noise eigenvalues
        noise_mask = e_val <= e_max
        if np.any(noise_mask):
            if shrinkage:
                e_val[noise_mask] = np.mean(e_val[noise_mask])
            else:
                e_val[noise_mask] = 0.0

        # Reconstruct clean correlation matrix
        corr_clean = e_vec @ np.diag(e_val) @ e_vec.T
        np.fill_diagonal(corr_clean, 1.0)

        # Convert back to covariance matrix
        cov_clean = corr_clean * np.outer(std, std)
        return cov_clean
