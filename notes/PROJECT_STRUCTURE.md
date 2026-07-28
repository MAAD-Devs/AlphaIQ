portfolio_dashboard/
│
├── .streamlit/
│   └── config.toml          # Streamlit theme and configuration
│
├── app.py                   # Main entry point for the Streamlit UI
│
├── core/                    # Core quantitative and financial logic
│   ├── __init__.py
│   ├── data_provider.py     # Fetches price histories (YFinance/AlphaVantage)
│   ├── data_models.py       # Pydantic or Dataclass schemas for portfolios
│   └── optimizer.py         # Riskfolio-Lib wrapper & drag adjustment logic
│
├── views/                   # UI Modules (Keeps app.py from being a giant file)
│   ├── __init__.py
│   ├── dashboard_view.py    # Displays current portfolio stats & risk metrics
│   └── optimization_view.py # Displays frontier plots and rebalancing options
│
├── requirements.txt         # Riskfolio-Lib, streamlit, yfinance, cvxpy, etc.
└── README.mdAGY
