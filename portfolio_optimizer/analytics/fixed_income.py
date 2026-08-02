"""
Fixed income analytics: Duration, Convexity, TIPS Inflation Breakeven, and Svensson (SVF) Yield Curve smoothing.
"""

from typing import List, Union

import numpy as np
from scipy.optimize import minimize


class BondAnalytics:
    """
    Computes core fixed income parameters: Price, Macaulay Duration, Modified Duration,
    Convexity, and TIPS Breakeven inflation rates.
    """

    @staticmethod
    def bond_price(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        years_to_maturity: float,
        freq: int = 2,
    ) -> float:
        """Calculates present value of a coupon-bearing bond."""
        periods = int(years_to_maturity * freq)
        c = (coupon_rate * face_value) / freq
        r = ytm / freq

        if r == 0:
            return float(c * periods + face_value)

        discount_factors = (1 + r) ** -np.arange(1, periods + 1)
        pv_coupons = np.sum(c * discount_factors)
        pv_principal = face_value * ((1 + r) ** -periods)
        return float(pv_coupons + pv_principal)

    @staticmethod
    def macaulay_duration(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        years_to_maturity: float,
        freq: int = 2,
    ) -> float:
        """Calculates Macaulay Duration in years."""
        price = BondAnalytics.bond_price(
            face_value, coupon_rate, ytm, years_to_maturity, freq
        )
        if price == 0:
            return 0.0

        periods = int(years_to_maturity * freq)
        c = (coupon_rate * face_value) / freq
        r = ytm / freq

        t_times = np.arange(1, periods + 1) / freq
        discount_factors = (1 + r) ** -np.arange(1, periods + 1)

        cash_flows = np.full(periods, c)
        cash_flows[-1] += face_value

        weighted_pv = np.sum(t_times * cash_flows * discount_factors)
        return float(weighted_pv / price)

    @staticmethod
    def modified_duration(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        years_to_maturity: float,
        freq: int = 2,
    ) -> float:
        """Calculates Modified Duration."""
        mac_dur = BondAnalytics.macaulay_duration(
            face_value, coupon_rate, ytm, years_to_maturity, freq
        )
        return float(mac_dur / (1 + (ytm / freq)))

    @staticmethod
    def convexity(
        face_value: float,
        coupon_rate: float,
        ytm: float,
        years_to_maturity: float,
        freq: int = 2,
    ) -> float:
        """Calculates Bond Convexity."""
        price = BondAnalytics.bond_price(
            face_value, coupon_rate, ytm, years_to_maturity, freq
        )
        if price == 0:
            return 0.0

        periods = int(years_to_maturity * freq)
        c = (coupon_rate * face_value) / freq
        r = ytm / freq

        t_times = np.arange(1, periods + 1) / freq
        discount_factors = (1 + r) ** -(np.arange(1, periods + 1) + 2)

        cash_flows = np.full(periods, c)
        cash_flows[-1] += face_value

        weighted_pv = np.sum(
            cash_flows * t_times * (t_times + 1 / freq) * discount_factors
        )
        return float(weighted_pv / price)

    @staticmethod
    def tips_breakeven(nominal_yield: float, real_tips_yield: float) -> float:
        """
        Calculates TIPS Breakeven Inflation Rate.
        Breakeven = (1 + Nominal) / (1 + Real) - 1
        """
        return float(((1 + nominal_yield) / (1 + real_tips_yield)) - 1.0)


class SVFYieldCurve:
    """
    Svensson (Sven-Svensson-Nelson-Siegel) Yield Curve Fitting & Smoothing.
    y(m) = beta0 + beta1 * ((1 - exp(-m/tau1)) / (m/tau1))
                 + beta2 * (((1 - exp(-m/tau1)) / (m/tau1)) - exp(-m/tau1))
                 + beta3 * (((1 - exp(-m/tau2)) / (m/tau2)) - exp(-m/tau2))
    """

    def __init__(
        self,
        beta0: float = 0.04,
        beta1: float = -0.01,
        beta2: float = 0.01,
        beta3: float = 0.005,
        tau1: float = 2.0,
        tau2: float = 5.0,
    ):
        self.params = [beta0, beta1, beta2, beta3, tau1, tau2]

    @staticmethod
    def yield_svf(
        m: Union[float, np.ndarray], params: List[float]
    ) -> Union[float, np.ndarray]:
        beta0, beta1, beta2, beta3, tau1, tau2 = params
        m = np.maximum(m, 1e-4)

        term1 = (1.0 - np.exp(-m / tau1)) / (m / tau1)
        term2 = term1 - np.exp(-m / tau1)
        term3 = ((1.0 - np.exp(-m / tau2)) / (m / tau2)) - np.exp(-m / tau2)

        return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

    def fit(self, maturities: np.ndarray, yields: np.ndarray) -> List[float]:
        """Fits Svensson model parameters to observed market yields."""

        def loss(p):
            y_pred = self.yield_svf(maturities, p)
            return np.sum((y_pred - yields) ** 2)

        initial_guess = [0.03, -0.01, 0.01, 0.0, 1.5, 5.0]
        bounds = [
            (-0.1, 0.2),
            (-0.2, 0.2),
            (-0.2, 0.2),
            (-0.2, 0.2),
            (0.1, 10.0),
            (0.1, 10.0),
        ]

        res = minimize(loss, initial_guess, bounds=bounds, method="L-BFGS-B")
        if res.success:
            self.params = list(res.x)
        return self.params

    def get_curve(self, maturities: np.ndarray) -> np.ndarray:
        """Returns smooth yield curve values for given maturities."""
        return np.asarray(self.yield_svf(maturities, self.params))
