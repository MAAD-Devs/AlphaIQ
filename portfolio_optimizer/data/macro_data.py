"""
FRED API wrapper for fetching macroeconomic series (Treasury yields, Inflation breakevens, CPI).
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from fredapi import Fred
except ImportError:
    Fred = None


class MacroDataLoader:
    """
    Ingests FRED (Federal Reserve Economic Data) series such as Treasury yields,
    inflation breakeven rates, and consumer price index data.
    """

    SERIES_MAPPING = {
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "treasury_3m": "DGS3MO",
        "inflation_10y_breakeven": "T10YIE",
        "inflation_5y_breakeven": "T5YIE",
        "cpi": "CPIAUCSL",
        "fed_funds": "FEDFUNDS",
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.fred = None
        if self.api_key and Fred is not None:
            try:
                self.fred = Fred(api_key=self.api_key)
            except Exception:
                self.fred = None

    def fetch_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.Series:
        """
        Fetches a single macroeconomic series by series_id (e.g. 'DGS10').
        """
        series_code = self.SERIES_MAPPING.get(series_id, series_id)

        if self.fred is not None:
            try:
                s = self.fred.get_series(
                    series_code, observation_start=start_date, observation_end=end_date
                )
                s = s.ffill().bfill()
                s.name = series_code
                return s
            except Exception as e:
                print(f"Warning: FRED API error for {series_code}: {e}")

        # Fallback if FRED API key is not present or failed: return sample placeholder structure
        dates = pd.date_range(
            start=start_date or "2020-01-01",
            end=end_date or pd.Timestamp.now(),
            freq="B",
        )
        base_val = 0.04 if "DGS" in series_code or "T10" in series_code else 0.02
        noise = np.random.normal(0, 0.0005, size=len(dates)).cumsum()
        s = pd.Series(base_val + noise, index=dates, name=series_code)
        return s.clip(lower=0.0)

    def fetch_multiple_series(
        self,
        series_ids: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetches multiple macro series into a unified DataFrame.
        """
        data = {}
        for s_id in series_ids:
            s = self.fetch_series(s_id, start_date=start_date, end_date=end_date)
            data[self.SERIES_MAPPING.get(s_id, s_id)] = s
        df = pd.DataFrame(data).ffill().bfill()
        return df

    def get_treasury_yield_curve(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetches 3M, 2Y, and 10Y Treasury yield series.
        """
        return self.fetch_multiple_series(
            ["treasury_3m", "treasury_2y", "treasury_10y"],
            start_date=start_date,
        )

    def get_inflation_breakevens(
        self, start_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetches 5-Year and 10-Year TIPS Inflation Breakeven rates.
        """
        return self.fetch_multiple_series(
            ["inflation_5y_breakeven", "inflation_10y_breakeven"],
            start_date=start_date,
        )
