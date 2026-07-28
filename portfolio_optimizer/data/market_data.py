"""
YFinance wrapper for downloading and processing historical price data for equities, ETFs, and REITs.
"""

from typing import List, Optional, Union
import pandas as pd
import numpy as np
import yfinance as yf


class MarketDataLoader:
    """
    Ingests market price data using yfinance and computes returns and price series.
    """

    def __init__(self, default_period: str = "5y", default_interval: str = "1d"):
        self.default_period = default_period
        self.default_interval = default_interval

    def fetch_prices(
        self,
        tickers: Union[str, List[str]],
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches adjusted close prices for the given list of tickers.
        """
        if isinstance(tickers, str):
            tickers = [tickers]

        period_to_use = period or self.default_period
        interval_to_use = interval or self.default_interval

        if start_date and end_date:
            df = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                interval=interval_to_use,
                auto_adjust=True,
                progress=False,
            )
        else:
            df = yf.download(
                tickers,
                period=period_to_use,
                interval=interval_to_use,
                auto_adjust=True,
                progress=False,
            )

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.levels[0]:
                prices = df["Close"]
            else:
                prices = df.xs(df.columns.levels[0][0], axis=1, level=0)
        else:
            prices = df[["Close"]] if "Close" in df.columns else df

        # Handle single ticker case
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        prices = prices.dropna(how="all").ffill().bfill()
        return prices

    def fetch_returns(
        self,
        tickers: Union[str, List[str]],
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        log_returns: bool = False,
    ) -> pd.DataFrame:
        """
        Calculates daily returns (simple or log) for specified tickers.
        """
        prices = self.fetch_prices(
            tickers=tickers,
            period=period,
            start_date=start_date,
            end_date=end_date,
        )
        if prices.empty:
            return pd.DataFrame()

        if log_returns:
            returns = np.log(prices / prices.shift(1)).dropna()
        else:
            returns = prices.pct_change().dropna()

        return returns

    def fetch_benchmark(
        self,
        benchmark_ticker: str = "^GSPC", # S&P500 as benchmark 
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """
        Fetches benchmark daily return series.
        """
        returns = self.fetch_returns(
            tickers=[benchmark_ticker],
            period=period,
            start_date=start_date,
            end_date=end_date,
        )
        if returns.empty:
            return pd.Series(dtype=float)
        return returns.iloc[:, 0]

if __name__ == "__main__":
    loader = MarketDataLoader(default_period="1mo")
    tickers = ["AAPL", "MSFT"]

    print("Testing MarketDataLoader...")

    # Test fetching prices
    print("\n--- Fetching Prices ---")
    prices = loader.fetch_prices(tickers)
    print(prices.head())
    print("Price shape:", prices.shape)

    # Test fetching returns
    print("\n--- Fetching Returns ---")
    returns = loader.fetch_returns(tickers, log_returns=False)
    print(returns.head())
    print("Returns shape:", returns.shape)

    # Test fetching benchmark
    print("\n--- Fetching Benchmark (^GSPC) ---")
    benchmark = loader.fetch_benchmark("^GSPC")
    print(benchmark.head())
    print("Benchmark shape:", benchmark.shape)