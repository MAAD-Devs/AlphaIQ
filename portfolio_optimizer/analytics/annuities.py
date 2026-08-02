"""
Annuity analytics: Participation/Cap payoffs and Monte Carlo option pricing.
"""

from typing import Dict, Optional

import numpy as np


class IndexedAnnuityPayoff:
    """
    Computes crediting payoffs for Fixed Indexed Annuities (FIA).
    Handles Point-to-Point indexing with Participation Rates, Caps, Floors, and Spreads.
    """

    @staticmethod
    def point_to_point_payoff(
        index_return: float,
        participation_rate: float = 1.0,
        cap_rate: Optional[float] = 0.10,
        floor_rate: float = 0.0,
        spread_rate: float = 0.0,
    ) -> float:
        """
        Calculates credited interest rate for an index period.
        Credited = min(Cap, max(Floor, (Index_Return * Part_Rate) - Spread))
        """
        effective_return = (index_return * participation_rate) - spread_rate
        if cap_rate is not None:
            effective_return = min(effective_return, cap_rate)
        credited_rate = max(floor_rate, effective_return)
        return float(credited_rate)


class AnnuityMonteCarloPricer:
    """
    Monte Carlo simulation engine for pricing indexed/variable annuity products and guaranteed benefit options.
    """

    def __init__(
        self,
        spot_price: float = 100.0,
        mu: float = 0.06,
        sigma: float = 0.15,
        risk_free_rate: float = 0.04,
    ):
        self.spot_price = spot_price
        self.mu = mu
        self.sigma = sigma
        self.risk_free_rate = risk_free_rate

    def simulate_paths(
        self, years: float = 10.0, steps_per_year: int = 252, n_simulations: int = 5000
    ) -> np.ndarray:
        """
        Simulates geometric Brownian motion price paths.
        Returns array of shape (n_simulations, steps + 1).
        """
        dt = 1.0 / steps_per_year
        n_steps = int(years * steps_per_year)

        np.random.seed(42)
        random_shocks = np.random.normal(0, 1, size=(n_simulations, n_steps))

        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * random_shocks

        log_returns = drift + diffusion
        cum_log_returns = np.cumsum(log_returns, axis=1)

        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = self.spot_price
        paths[:, 1:] = self.spot_price * np.exp(cum_log_returns)

        return paths

    def price_fia_contract(
        self,
        years: int = 10,
        participation_rate: float = 1.0,
        cap_rate: Optional[float] = 0.08,
        floor_rate: float = 0.0,
        n_simulations: int = 5000,
    ) -> Dict[str, float]:
        """
        Prices a 1-year annual reset Fixed Indexed Annuity contract over a multi-year horizon.
        """
        paths = self.simulate_paths(
            years=years, steps_per_year=252, n_simulations=n_simulations
        )
        # Annual price checkpoints
        annual_indices = [int(i * 252) for i in range(years + 1)]
        annual_prices = paths[:, annual_indices]

        # Calculate annual index returns
        annual_returns = (annual_prices[:, 1:] - annual_prices[:, :-1]) / annual_prices[
            :, :-1
        ]

        # Apply crediting strategy for each year
        credited_returns = np.zeros_like(annual_returns)
        for sim in range(n_simulations):
            for t in range(years):
                r = annual_returns[sim, t]
                credited_returns[sim, t] = IndexedAnnuityPayoff.point_to_point_payoff(
                    index_return=r,
                    participation_rate=participation_rate,
                    cap_rate=cap_rate,
                    floor_rate=floor_rate,
                )

        # Compound credited growth
        account_values = self.spot_price * np.prod(1 + credited_returns, axis=1)
        discount_factor = np.exp(-self.risk_free_rate * years)
        pv_account_values = account_values * discount_factor

        return {
            "expected_final_value": float(np.mean(account_values)),
            "pv_expected_value": float(np.mean(pv_account_values)),
            "5th_percentile_value": float(np.percentile(account_values, 5)),
            "95th_percentile_value": float(np.percentile(account_values, 95)),
            "mean_annualized_credited_return": float(
                np.mean(np.mean(credited_returns, axis=1))
            ),
        }
