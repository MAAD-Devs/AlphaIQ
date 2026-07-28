"""
Domain models for assets, portfolios, market data requests, and optimization results.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class AssetClass(str, Enum):
    EQUITY = "Equity"
    ETF = "ETF"
    REIT = "REIT"
    BOND = "Bond"
    ANNUITY = "Annuity"
    CASH = "Cash"


AssetType = AssetClass  # Alias for backward compatibility


@dataclass(frozen=True)
class Asset:
    """
    Domain object representing an individual investment asset.
    """

    ticker: str
    asset_class: AssetClass
    name: str = ""
    annual_drag: float = 0.0  # Annualized fee drag (e.g. ETF expense ratio)


@dataclass
class Portfolio:
    """
    Domain object representing a portfolio of assets.
    """

    name: str
    asset_values: Dict[Asset, float]  # Dictionary of assets and their dollar values
    account_drag: float = 0.0

    @property
    def total_value(self) -> float:
        """Calculates total portfolio dollar value"""
        return sum(self.asset_values.values())

    @property
    def tickers(self) -> List[str]:
        """Returns a list of all asset tickers"""
        return [asset.ticker for asset in self.asset_values]

    @property
    def weights(self) -> Dict[str, float]:
        """Dynamically computes percentage weight per ticker based on asset values."""
        total = self.total_value
        if total == 0:
            return {asset.ticker: 0.0 for asset in self.asset_values}
        return {asset.ticker: value / total for asset, value in self.asset_values.items()}


@dataclass
class MarketDataRequest:
    """
    Domain object representing a market data fetch request.
    """

    tickers: List[str]
    period: str = "5y"
    interval: str = "1d"
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class OptimizationResult:
    """
    Domain object representing the output of a portfolio optimization solver.
    """

    method: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    status: str = "Optimal"
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_portfolio(
        self,
        name: str,
        total_value: float,
        asset_map: Optional[Dict[str, Asset]] = None,
        account_drag: float = 0.0,
    ) -> Portfolio:
        """
        Constructs a Portfolio domain object based on optimized weights and total dollar value.
        """
        asset_values = {}
        for ticker, weight in self.weights.items():
            val = total_value * weight
            if asset_map and ticker in asset_map:
                asset_obj = asset_map[ticker]
            else:
                asset_obj = Asset(ticker=ticker, asset_class=AssetClass.EQUITY)
            asset_values[asset_obj] = val

        return Portfolio(
            name=name,
            asset_values=asset_values,
            account_drag=account_drag,
        )


if __name__ == "__main__":
    aapl = Asset(ticker="AAPL", asset_class=AssetClass.EQUITY, name="Apple Inc.", annual_drag=0.0)
    vti = Asset(ticker="VTI", asset_class=AssetClass.ETF, name="Vanguard Total Stock Market ETF", annual_drag=0.0003)

    portfolio = Portfolio(
        name="Growth Portfolio",
        asset_values={aapl: 30000.0, vti: 70000.0},
        account_drag=0.0015,
    )

    print(f"Total Value: ${portfolio.total_value:,.2f}")
    print(f"Weights: {portfolio.weights}")
