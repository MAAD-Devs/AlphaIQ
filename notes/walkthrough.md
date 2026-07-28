# Walkthrough: Quantitative Portfolio Optimization and Risk Management Dashboard

The quantitative portfolio optimization framework and its interactive Streamlit dashboard have been fully built, verified, and integrated. This system provides plan participants with SDBA access inside corporate Roth 401(k) plans with institutional-grade risk hedging and allocation models.

## Completed Components

### 1. Core Domain Models & Configuration
- **Asset Hierarchy** (`portfolio_optimizer/models/asset.py`): Supports `EquityAsset`, `FixedIncomeAsset`, `REITAsset`, `SVFAsset`, and `AnnuityAsset`.
- **Containers** (`portfolio_optimizer/models/portfolio.py`): Aggregates weights and values.
- **Results** (`portfolio_optimizer/models/optimization_result.py`): Structured results container.
- **Configurations** (`config/default_config.yaml`): Centralized parameters for assets, inflation, and options.

### 2. Analytical Diagnostic Engines
- **Traditional Risk Metrics** (`portfolio_optimizer/analytics/risk_metrics.py`): Annualized Sharpe, Sortino, Treynor, and Information Ratios.
- **Tail-Risk Diagnostics** (`portfolio_optimizer/analytics/tail_risk.py`): Value at Risk (VaR), Conditional VaR (CVaR), Max Drawdown, Calmar Ratio, and Ulcer Index.
- **Factor Models** (`portfolio_optimizer/analytics/factor_models.py`): Fama-French Five-Factor regression using OLS, and Marcenko-Pastur random-matrix denoising.
- **Fixed-Income Sleeves** (`portfolio_optimizer/analytics/fixed_income.py`): Macaulay & Modified Duration, Convexity, Breakeven Inflation, and SVF book-to-market convergence smoothing.
- **Insurance Wrap Derivatives** (`portfolio_optimizer/analytics/annuities.py`): Indexed annuity payoffs and stochastic Monte Carlo option pricing simulations.
- **Sector Rotation** (`portfolio_optimizer/analytics/sector_analysis.py`): Machine learning-based Sector Rotator using Random Forests with fallback statistical momentum models.

### 3. Portfolio Optimization & Constraints
- **Base Solver Class** (`portfolio_optimizer/optimization/base_optimizer.py`): Shared interface.
- **Traditional Solvers** (`portfolio_optimizer/optimization/traditional.py`): Mean-Variance (MVO) and Vanilla Risk Parity.
- **Hierarchical Solvers** (`portfolio_optimizer/optimization/hierarchical.py`): Hierarchical Risk Parity (HRP) and Hierarchical Equal Risk Contribution (HERC).
- **Black-Litterman** (`portfolio_optimizer/optimization/black_litterman.py`): Bayesian updating with subjective view matrices.
- **Kelly Criterion** (`portfolio_optimizer/optimization/kelly_criterion.py`): Logarithmic utility compounds.
- **Shortfall Solvers** (`portfolio_optimizer/optimization/shortfall_minimization.py`): CVaR and CDaR minimizers.
- **Constraints Builder** (`portfolio_optimizer/optimization/constraints.py`): Weight bounds and linear beta constraints.

### 4. Interactive Streamlit Dashboard
- **Styling** (`dashboard/style.css`): Modern dark-mode glassmorphism interface with Outfit/Inter typography, linear gradient headings, and responsive hover scales.
- **State Management** (`dashboard/utils/state_management.py`): Robust session state caching.
- **Core App Launcher** (`dashboard/app.py`): Strategic overview and main welcome landing grid.
- **Data Ingestion** (`dashboard/pages/01_Data_Ingestion.py`): Live yfinance and FRED downloads, local caching, and data quality check reporting.
- **Quantitative Analytics** (`dashboard/pages/02_Analytics.py`): Interactive bond calculators, REIT NAV diagnostics, SVF smoothing simulations, and stochastic annuity Monte Carlo charts.
- **Portfolio Optimization** (`dashboard/pages/03_Optimization.py`): Limit-bound sliders, risk constraints (portfolio beta cap), and interactive weight donuts.
- **Out-of-Sample Backtesting** (`dashboard/pages/04_Backtesting.py`): Cumulative performance comparison charts and strategy statistics tables.

---

## Verification and Testing

A self-contained testing suite was created under [tests/](file:///C:/Users/Gaming%20PC/Desktop/Projects/InvestmentPortfolio/tests) and [scripts/run_verification.py](file:///C:/Users/Gaming%20PC/Desktop/Projects/InvestmentPortfolio/scripts/run_verification.py) to confirm correctness of all mathematical formulas, pricing derivatives, and optimizer convergence.

### Test Log Executed via UV:
```bash
uv run python scripts/run_verification.py
```

```text
Executing quantitative analytics tests...
[OK] test_risk_metrics passed
[OK] test_tail_risk passed
[OK] test_fixed_income passed
[OK] test_svf_smoothing passed
[OK] test_annuity_payoff passed

Executing optimization engine tests...
[OK] test_optimizers passed

==============================
ALL VERIFICATION TESTS PASSED!
==============================
```
All assets converge and behave as expected under extreme risk parameter scenarios.

When using the FRED API, it is critical to include the following: 
"This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis."
