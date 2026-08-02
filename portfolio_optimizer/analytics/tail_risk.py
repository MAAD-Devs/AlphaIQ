"""
Tail risk and drawdown analytics: VaR, CVaR, Max Drawdown, Calmar Ratio, and Ulcer Index.
"""

from typing import Union

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew


def ValueAtRisk(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95,
    method: str = "historical",
) -> float:
    """
    Calculates Value at Risk (VaR) at specified confidence level.
    Methods: 'historical', 'parametric', 'cornish_fisher'.
    Returns a positive float representing potential percentage loss.
    """
    returns_arr = np.asarray(returns)
    if len(returns_arr) == 0:
        return 0.0

    alpha = 1.0 - confidence_level

    if method == "historical":
        var_val = -np.percentile(returns_arr, alpha * 100)
    elif method == "parametric":
        mu = np.mean(returns_arr)
        sigma = np.std(returns_arr, ddof=1)
        z = norm.ppf(confidence_level)
        var_val = z * sigma - mu
    elif method == "cornish_fisher":
        mu = np.mean(returns_arr)
        sigma = np.std(returns_arr, ddof=1)
        s = skew(returns_arr)
        k = kurtosis(returns_arr) - 3  # excess kurtosis
        z = norm.ppf(confidence_level)

        # Cornish-Fisher expansion adjusted quantile
        z_cf = (
            z
            + (s / 6) * (z**2 - 1)
            + (k / 24) * (z**3 - 3 * z)
            - (s**2 / 36) * (2 * z**3 - 5 * z)
        )
        var_val = z_cf * sigma - mu
    else:
        raise ValueError(f"Unknown VaR method: {method}")

    return float(max(0.0, var_val))


def ConditionalVaR(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.95,
) -> float:
    """
    Calculates Conditional Value at Risk (CVaR / Expected Shortfall).
    CVaR is the expected loss given that loss exceeds VaR.
    """
    returns_arr = np.asarray(returns)
    if len(returns_arr) == 0:
        return 0.0

    alpha = 1.0 - confidence_level
    cutoff_quantile = np.percentile(returns_arr, alpha * 100)
    tail_losses = returns_arr[returns_arr <= cutoff_quantile]

    if len(tail_losses) == 0:
        return 0.0

    cvar_val = -np.mean(tail_losses)
    return float(max(0.0, cvar_val))


def MaximumDrawdown(
    prices_or_returns: Union[pd.Series, np.ndarray], is_returns: bool = True
) -> float:
    """
    Calculates Maximum Drawdown (mdd).
    Returns a positive float representing peak-to-trough decline (e.g. 0.25 for 25%).
    """
    arr = np.asarray(prices_or_returns)
    if len(arr) == 0:
        return 0.0

    if is_returns:
        cum_returns = np.cumprod(1 + arr)
        prices = np.insert(cum_returns, 0, 1.0)
    else:
        prices = arr

    running_max = np.maximum.accumulate(prices)
    drawdowns = (running_max - prices) / running_max
    mdd = np.max(drawdowns)
    return float(mdd)


def CalmarRatio(
    returns: Union[pd.Series, np.ndarray],
    annualization_factor: int = 252,
) -> float:
    """
    Calculates Calmar Ratio = Annualized Return / Maximum Drawdown.
    """
    returns_arr = np.asarray(returns)
    if len(returns_arr) == 0:
        return 0.0

    annual_return = np.mean(returns_arr) * annualization_factor
    mdd = MaximumDrawdown(returns_arr, is_returns=True)

    if mdd == 0 or np.isnan(mdd):
        return 0.0

    return float(annual_return / mdd)


def UlcerIndex(
    prices_or_returns: Union[pd.Series, np.ndarray], is_returns: bool = True
) -> float:
    """
    Calculates Ulcer Index measuring downside risk stress over time.
    UI = sqrt( mean( % drawdown ^ 2 ) )
    """
    arr = np.asarray(prices_or_returns)
    if len(arr) == 0:
        return 0.0

    if is_returns:
        cum_returns = np.cumprod(1 + arr)
        prices = np.insert(cum_returns, 0, 1.0)
    else:
        prices = arr

    running_max = np.maximum.accumulate(prices)
    pct_drawdowns = ((prices - running_max) / running_max) * 100
    squared_drawdowns = pct_drawdowns**2
    ui = np.sqrt(np.mean(squared_drawdowns))
    return float(ui)
