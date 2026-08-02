# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Branch Rules

- **Never push directly to `release`** — it is the production branch deployed to Streamlit Cloud. All changes go through `master` (or a feature branch) first, then are merged into `release` via PR.

## Running the App

```bash
python -m streamlit run app.py
```

For verbose logging (DEBUG and above), set `APP_ENV=TEST` — either in `.env` or inline:

```bash
APP_ENV=TEST python -m streamlit run app.py
```

Without `APP_ENV=TEST`, app loggers emit WARNING and above only. Errors still surface but no PII appears in log output.

The app runs on port 8501. All pages auto-discover from the `pages/` directory via Streamlit's multipage convention.

## Development Setup

```bash
pip install -r requirements.txt
```

Python version: **3.14** (see `.python-version`). The devcontainer uses Python 3.11.

## Adding New Packages

Streamlit Cloud deploys from `uv.lock`, not `requirements.txt`. Adding a package to `requirements.txt` alone won't be picked up in production. Always do all three:

```bash
python -m uv add "<package>"      # updates uv.lock and pyproject.toml
pip install "<package>"           # installs into the system Python used locally
```

Then manually add the same package to `requirements.txt`.

> **Note**: The app runs against the system Python, not the uv venv. Both `uv add` and `pip install` are required when adding packages — `uv add` alone will cause `ModuleNotFoundError` at runtime locally.

## Database Migrations

Migrations are managed with [dbmate](https://github.com/amacneil/dbmate). Migration files live in `db/migrations/`.

### Setup

1. Install dbmate:
   - **Mac**: `brew install dbmate`
   - **Windows**: download the binary from [github.com/amacneil/dbmate/releases](https://github.com/amacneil/dbmate/releases) and add it to your PATH
   - **Docker**: `docker run --rm ghcr.io/amacneil/dbmate`

2. Copy `.env.example` to `.env` and fill in the Supabase direct connection string:
   - Supabase dashboard → Settings → Database → Connection string → **Direct connection**

### Common commands

```bash
dbmate up          # apply all pending migrations
dbmate down        # roll back the last migration
dbmate status      # show which migrations have/haven't been applied
dbmate new <name>  # create a new migration file
```

### Writing migrations

Each file in `db/migrations/` has two sections:

```sql
-- migrate:up
CREATE TABLE ...

-- migrate:down
DROP TABLE ...
```

Always write a `-- migrate:down` block. It keeps rollbacks cheap and is required for `dbmate down` to work.

### Deploying schema changes

Run `dbmate up` against the production Supabase database before merging code that depends on the new schema. The `schema_migrations` table in Supabase tracks which migrations have been applied.

## Authentication

The app uses Streamlit's built-in Google OAuth (`st.user`, `st.login`, `st.logout`). Auth is gated in `app.py` before any page content renders.

Configuration lives in `.streamlit/secrets.toml` (gitignored). Each dev needs their own copy with the shared credentials. Required keys:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "<random 32-byte hex>"

[auth.google]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://accounts.google.com/.well-known/openid-configuration"

[supabase]
url = "https://<project-ref>.supabase.co"
key = "<publishable anon key>"
```

The Supabase publishable key is found in the Supabase dashboard under **Settings → API**. The **Data API must be enabled** (Settings → API) for `supabase-py` to work.

## Architecture

### Multipage Streamlit App

- `app.py` — Executive overview / landing page. Also owns the shared sidebar (risk-free rate, lookback period, market data refresh).
- `pages/01_Data_Ingestion.py` — Portfolio holdings editor (tickers, quantities, fee drag).
- `pages/02_Analytics.py` — Risk ratios (Sharpe, Sortino, Treynor), tail risk (VaR/CVaR), factor models, fixed income.
- `pages/03_Optimization.py` — Runs optimization solvers and displays results.
- `pages/04_Backtesting.py` — Out-of-sample backtest with rebalancing frequency selection.

### Shared State (`utils/state_management.py`)

All pages share state through `st.session_state`. The keys are:
- `portfolio` — active `Portfolio` domain object
- `returns_df` / `prices_df` — cached historical return/price DataFrames (fetched via `fetch_and_cache_market_data`)
- `benchmark_series` — S&P 500 benchmark daily returns
- `risk_free_rate`, `lookback_period`, `benchmark_ticker`

Every page calls `init_session_state()` and `inject_custom_css()` at the top. `fetch_and_cache_market_data()` falls back to synthetic data if the yfinance fetch fails, so the app remains runnable offline.

### Database Layer (`utils/db.py`)

Thin wrapper around the Supabase client. Three functions:
- `get_or_create_user(email)` — upserts a row in `users` on first login
- `load_active_portfolio(email)` — reads the user's saved portfolio from `portfolios` + `portfolio_assets` and returns a `Portfolio`, or `None` if none exists
- `save_portfolio(email, portfolio)` — upserts the portfolio row and replaces all asset rows

`init_session_state()` calls `get_or_create_user` and `load_active_portfolio` on first load (when `portfolio` is not yet in session state). The "Save Portfolio Changes" button in `01_Data_Ingestion.py` calls `save_portfolio`. Both sites log errors via `logging` and fall back gracefully if the DB is unavailable.

### Core Package (`portfolio_optimizer/`)

**Domain Models** (`core/data_models.py`):
- `Asset(ticker, asset_class, name, annual_drag)` — frozen dataclass
- `Portfolio(name, asset_values: Dict[Asset, float], account_drag)` — weights computed dynamically from dollar values
- `OptimizationResult` — returned by all solvers; has `.to_portfolio()` to convert back to a `Portfolio`

**Optimization Solvers** (`optimization/`): All solvers extend `BasePortfolioOptimizer` and implement `optimize(returns, ...) -> OptimizationResult`. Available solvers:
- `MeanVarianceOptimizer`, `RiskParityOptimizer` (traditional.py)
- `BlackLittermanOptimizer` (uses S&P 500 equilibrium as market prior)
- `HierarchicalRiskParityOptimizer` (hierarchical.py)
- `KellyCriterionOptimizer`
- `ShortfallMinimizationOptimizer` (CVaR/CDaR minimization)

`BasePortfolioOptimizer.compute_summary_stats()` is the shared method for computing annualized return, volatility, and Sharpe ratio across all solvers.

**Analytics** (`analytics/`): Standalone functions (not classes) for Sharpe, Sortino, Treynor, VaR, CVaR, max drawdown, factor models, fixed income duration, and sector analysis. All operate on `pd.Series`/`pd.DataFrame` of daily returns.

**Data** (`data/`): `MarketDataLoader` wraps yfinance. `macro_data.py` uses the FRED API (requires `FRED_API_KEY` in `.streamlit/secrets.toml`). `fundamentals.py` uses `financetoolkit`.

### Preset Portfolio Templates

Three presets are defined in `utils/state_management.py` (`PRESET_TEMPLATES`): "Growth & Tech", "60/40 Retirement", "Multi-Asset Endowment". Templates are loaded into `st.session_state.portfolio` via `load_portfolio_template()`.

## Key Conventions

- `Portfolio.weights` is computed dynamically from `asset_values` dollar amounts — never stored directly.
- Annualization uses 252 trading days throughout.
- Fee drag is additive: optimizer base drag + portfolio `account_drag` + weighted asset `annual_drag` values.
- Each page sets `st.set_page_config()` independently (required by Streamlit multipage architecture).
