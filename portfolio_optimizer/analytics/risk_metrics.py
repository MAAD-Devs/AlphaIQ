"""
Specialized risk ratio analytics: Sharpe, Sortino, Treynor, and Information Ratios.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd


def SharpeRatio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.04,
    annualization_factor: int = 252,
) -> float:
    """
    Computes annualized Sharpe Ratio.
    Sharpe = (E[R] - R_f) / std(R)
    """
    returns_arr = np.asarray(returns)
    if len(returns_arr) == 0:
        return 0.0

    rf_daily = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    excess_returns = returns_arr - rf_daily
    std = np.std(excess_returns, ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0

    mean_excess = np.mean(excess_returns)
    return float((mean_excess / std) * np.sqrt(annualization_factor))


def SortinoRatio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.04,
    target_return: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """
    Computes annualized Sortino Ratio considering downside volatility.
    Sortino = (E[R] - R_f) / downside_std
    """
    returns_arr = np.asarray(returns)
    if len(returns_arr) == 0:
        return 0.0

    rf_daily = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    target_daily = (1 + target_return) ** (1 / annualization_factor) - 1

    downside_diff = np.minimum(0, returns_arr - target_daily)
    downside_std = np.sqrt(np.mean(downside_diff**2))

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    mean_excess = np.mean(returns_arr - rf_daily)
    return float((mean_excess / downside_std) * np.sqrt(annualization_factor))


def TreynorRatio(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.04,
    annualization_factor: int = 252,
) -> float:
    """
    Computes annualized Treynor Ratio.
    Treynor = (E[R] - R_f) / Beta
    """
    r_arr = np.asarray(returns)
    b_arr = np.asarray(benchmark_returns)

    if len(r_arr) == 0 or len(r_arr) != len(b_arr):
        return 0.0

    cov_matrix = np.cov(r_arr, b_arr)
    cov = cov_matrix[0, 1]
    var_b = cov_matrix[1, 1]

    if var_b == 0 or np.isnan(var_b):
        return 0.0

    beta = cov / var_b
    if beta == 0 or np.isnan(beta):
        return 0.0

    rf_daily = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    mean_excess_annual = np.mean(r_arr - rf_daily) * annualization_factor

    return float(mean_excess_annual / beta)


def InformationRatio(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    annualization_factor: int = 252,
) -> float:
    """
    Computes annualized Information Ratio.
    IR = (E[R] - E[R_b]) / Tracking_Error
    """
    r_arr = np.asarray(returns)
    b_arr = np.asarray(benchmark_returns)

    if len(r_arr) == 0 or len(r_arr) != len(b_arr):
        return 0.0

    diff = r_arr - b_arr
    tracking_error = np.std(diff, ddof=1)

    if tracking_error == 0 or np.isnan(tracking_error):
        return 0.0

    mean_diff = np.mean(diff)
    return float((mean_diff / tracking_error) * np.sqrt(annualization_factor))


def compute_all_risk_metrics(
    returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Optional[Union[pd.Series, np.ndarray]] = None,
    risk_free_rate: float = 0.04,
    annualization_factor: int = 252,
) -> Dict[str, float]:
    """
    Helper function to calculate all risk-adjusted return ratios in one call.
    """
    metrics = {
        "sharpe_ratio": SharpeRatio(returns, risk_free_rate, annualization_factor),
        "sortino_ratio": SortinoRatio(
            returns, risk_free_rate, 0.0, annualization_factor
        ),
    }

    if benchmark_returns is not None:
        metrics["treynor_ratio"] = TreynorRatio(
            returns, benchmark_returns, risk_free_rate, annualization_factor
        )
        metrics["information_ratio"] = InformationRatio(
            returns, benchmark_returns, annualization_factor
        )

    return metrics
