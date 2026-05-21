# PortfolioOptimizer

An advanced quantitative portfolio optimization and risk management system with a Streamlit web interface. Provides institutional-grade tools for portfolio construction, risk analytics, stress testing, factor analysis, and PDF reporting.

> **Disclaimer:** This tool is a calculation-assistance aid. It is not intended as the sole basis for financial decisions. Always do your own research (DYOR).

---

## Quick Start

### 1. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```env
ALPHA_VANTAGE_KEY=your_key_here
TWELVE_DATA_KEY=your_key_here
FMP_KEY=your_key_here
```

yfinance (the primary data source) works without any key. The other sources are fallbacks.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the web interface

```bash
streamlit run app/Home.py
```

### 4. Python API

```python
from src.data.data_manager import DataManager
from src.optimization.optimizers import PortfolioOptimizer

dm = DataManager()
prices = dm.get_price_data(['AAPL', 'GOOGL', 'MSFT'], '2020-01-01', '2024-01-01')
returns = dm.get_returns(['AAPL', 'GOOGL', 'MSFT'], '2020-01-01', '2024-01-01')

optimizer = PortfolioOptimizer(returns)
result = optimizer.optimize(method='max_sharpe')
print(result['weights'])
```

### 5. Standalone stock valuation

```bash
python stock_valuer.py
```

---

## Features

| Capability | Description |
|---|---|
| Data Ingestion | Multi-source fallback: yfinance → Alpha Vantage → Twelve Data → FMP, with TTL caching |
| Portfolio Optimization | 7 algorithms: Max Sharpe, Min Vol, Risk Parity, HRP, Max Diversification, Max Return, Equal Weight |
| Risk Analytics | VaR/CVaR (3 methods), GARCH, drawdown family, Sharpe/Sortino/Calmar/Omega |
| Deep Risk Analysis | Enhanced VaR comparison, tail risk (JB test, QQ plot), Monte Carlo, component VaR, ENB via PCA |
| Hedging Effectiveness | Beta classification, risk contribution decomposition, diversification benefit waterfall |
| Stress Testing | 7 historical crisis scenarios + parametric + Monte Carlo (4 simulation methods) |
| Factor Analysis | Fama-French 3/5-factor, style factors (momentum/value/quality/low-vol/size), Brinson attribution |
| Portfolio Tracking | Holdings entry (tickers + shares), diversity metrics (HHI, Gini, effective stocks) |
| Rebalancing | Drift detection, trade recommendations with share counts, DCA scheduler |
| Reporting | PDF generation with 6 chart types and 3 report templates |
| Stock Valuation | Standalone 3-stage pipeline: Magic Formula screen → Multi-Factor Score → Reverse DCF |

---

## Repository Layout

```
PortfolioOptimizer/
│
├── app/                                  # Streamlit multi-page application
│   ├── Home.py                           # Landing page, session state init
│   └── pages/
│       ├── 1_Portfolio_Input.py          # Holdings entry and data loading
│       ├── 2_Optimization.py             # Optimization + rebalancing UI
│       ├── 3_Risk_Analytics.py           # Risk dashboard + deep analysis
│       ├── 4_Stress_Testing.py           # Historical scenarios + Monte Carlo
│       ├── 5_Monitoring.py               # Rebalancing + attribution + DCA
│       ├── 6_Factor_Analysis.py          # FF factors + style + decomposition
│       └── 7_Reports.py                  # PDF report generation UI
│
├── src/                                  # Core library
│   ├── data/
│   │   ├── data_manager.py               # Orchestrated multi-source fetcher (448 lines)
│   │   ├── sources.py                    # YFinance/AV/TwelveData/FMP adapters (521 lines)
│   │   ├── cache.py                      # TTL-based disk cache
│   │   └── validators.py                 # OHLCV data validation utilities
│   │
│   ├── optimization/
│   │   ├── optimizers.py                 # PortfolioOptimizer — 7 methods (518 lines)
│   │   ├── constraints.py                # PortfolioConstraints dataclass
│   │   ├── black_litterman.py            # Black-Litterman model (432 lines)
│   │   └── hrp.py                        # Standalone HRP implementation
│   │
│   ├── risk/
│   │   ├── metrics.py                    # RiskMetrics — 20+ metrics (673 lines)
│   │   ├── var.py                        # VaRCalculator: historical/param/MC (559 lines)
│   │   └── garch.py                      # GARCHModel + ewma_volatility (495 lines)
│   │
│   ├── portfolio/
│   │   ├── calculator.py                 # PositionCalculator — share sizing (430 lines)
│   │   ├── holdings.py                   # HoldingsTracker + diversity metrics
│   │   └── rebalancer.py                 # Rebalancer + DCAScheduler + Attribution (489 lines)
│   │
│   ├── factors/
│   │   ├── fama_french.py                # FamaFrenchAnalyzer 3/5-factor (436 lines)
│   │   ├── attribution.py                # Brinson attribution + sector analysis (445 lines)
│   │   ├── style.py                      # StyleFactorAnalyzer (474 lines)
│   │   └── decomposition.py              # FactorRiskDecomposition (399 lines)
│   │
│   ├── simulation/
│   │   ├── monte_carlo.py                # MonteCarloSimulator — 4 methods (547 lines)
│   │   └── scenarios.py                  # StressTester + HISTORICAL_SCENARIOS (527 lines)
│   │
│   ├── reports/
│   │   ├── generator.py                  # ReportGenerator (reportlab PDF)
│   │   ├── templates.py                  # 3 report templates (513 lines)
│   │   └── charts.py                     # ReportChartGenerator — 6 chart types (501 lines)
│   │
│   └── utils/
│       ├── helpers.py                    # validate_tickers, format_*, annualize_*
│       └── logger.py                     # loguru-based structured logging
│
├── stock_valuer.py                       # Standalone 3-stage valuation pipeline (1,295 lines)
├── requirements.txt                      # Python dependencies
├── setup.py                              # Package configuration
├── .env.example                          # API key template
└── README.md                             # This file
```

**Scale:** ~19,100 lines across 44 Python files.

---

## Architecture

### Technology Stack

#### Core Compute
| Package | Version | Role |
|---|---|---|
| pandas | >=2.0.0 | DataFrames, time-series operations |
| numpy | >=1.24.0 | Array math, matrix operations |
| scipy | >=1.10.0 | SLSQP optimizer, hierarchical clustering, stats |

#### Financial Data
| Package | Version | Role |
|---|---|---|
| yfinance | >=0.2.40 | Primary OHLCV + fundamentals |
| alpha_vantage | >=2.3.1 | Fallback price data (500 req/day) |
| pandas_datareader | >=0.10.0 | Fama-French factor data from Ken French |
| requests | >=2.31.0 | Twelve Data + FMP REST APIs |

#### Optimization & Risk
| Package | Version | Role |
|---|---|---|
| arch | >=6.3.0 | GARCH(1,1) volatility modeling |
| scikit-learn | >=1.3.0 | PCA for Effective Number of Bets |
| statsmodels | >=0.14.0 | OLS for Fama-French regressions |

#### Web & Visualization
| Package | Version | Role |
|---|---|---|
| streamlit | >=1.31.0 | Web interface framework |
| plotly | >=5.18.0 | All interactive charts |
| matplotlib | >=3.7.0 | Static charts for PDF reports |
| reportlab | >=4.0.0 | PDF generation |

#### Utilities
| Package | Version | Role |
|---|---|---|
| loguru | >=0.7.0 | Structured logging with rotation |
| tenacity | >=8.2.3 | Exponential-backoff retry for network calls |
| python-dotenv | >=1.0.0 | API key management |
| rich | >=13.0.0 | Terminal formatting for stock_valuer.py |
| tabulate | >=0.9.0 | Table formatting fallback |

---

### Data Layer (`src/data/`)

**`DataManager`** is the central orchestrator. It implements a priority-ordered fallback chain across 4 data sources with TTL caching.

```
get_price_data(tickers, start_date, end_date)
    → Check DataCache  (86400s TTL for historical, 3600s for intraday)
    → Try YFinanceSource     (primary, batched in groups of 5, 3 retries)
    → Try AlphaVantageSource (500 req/day, 12s rate limit)
    → Try TwelveDataSource   (800 req/day, 8s rate limit)
    → Try FMPSource          (0.5s between calls)
    → Validate with DataValidator
    → Cache + return prices DataFrame (dates x tickers)
```

Key methods: `get_price_data`, `get_returns`, `get_ohlcv_data`, `get_current_prices`, `get_sector_info`, `clear_cache`, `get_cache_stats`.

---

### Optimization Engine (`src/optimization/`)

All numeric optimization uses `scipy.optimize.minimize` (SLSQP). The constructor pre-computes annualized mean returns and covariance matrix once.

```python
PortfolioOptimizer(returns: pd.DataFrame, risk_free_rate=0.02, frequency=252)
result = optimizer.optimize(method='max_sharpe', constraints=constraints)
# result = {weights, expected_return, volatility, sharpe_ratio, method}
```

| Method | Algorithm | Objective |
|---|---|---|
| `max_sharpe` | SLSQP | Maximize (mu - rf) / sigma |
| `min_volatility` | SLSQP | Minimize sqrt(w'Sw) |
| `max_return` | SLSQP | Maximize w·mu |
| `max_diversification` | SLSQP | Maximize weighted-avg vol / portfolio vol |
| `risk_parity` | SLSQP | Minimize sum((RC_i - RC_target)^2) |
| `hrp` | Hierarchical clustering | Recursive bisection, inverse-variance weights |
| `equal_weight` | Analytical | 1/n for all assets |

**HRP detail:** correlation-based distance d_ij = sqrt(0.5*(1-rho_ij)), single-linkage clustering, quasi-diagonal reordering, recursive bisection with inverse-variance cluster allocation.

**Efficient Frontier:** sweeps target returns from min-vol to max-return; per point uses SLSQP with two equality constraints (sum(w)=1, w·mu=target); options ftol=1e-9, maxiter=1000; accepts near-converged solutions when |sum(w)-1| < 1e-3; output sorted by volatility.

**`black_litterman.py`** (432 lines): full Black-Litterman model with market equilibrium prior + investor views (P, Q matrices). Implemented but not yet wired into the UI.

---

### Risk Analytics (`src/risk/`)

**`RiskMetrics`** provides 20+ metrics:

```python
RiskMetrics(returns: pd.DataFrame, risk_free_rate=0.02, frequency=252)
```

- Volatility: `historical_volatility`, `parkinson_volatility` (range-based), `downside_deviation`
- Ratios: `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `omega_ratio`
- Drawdown: `max_drawdown`, `average_drawdown`, `ulcer_index`, `drawdown_series`
- Summary: `summary_table(prices)` — all metrics in one DataFrame

**`VaRCalculator`:** `historical_var`, `parametric_var`, `monte_carlo_var`, `calculate_all`

**`GARCHModel`:** GARCH(1,1) via `arch` library. `ewma_volatility(returns, decay=0.94)` for EWMA vol.

---

### Portfolio Management (`src/portfolio/`)

**`HoldingsTracker`:** takes `{ticker: num_shares}` + current prices; computes HHI, effective stock count, Gini coefficient, diversification ratio, concentration metrics; provides text recommendations.

**`PositionCalculator`:** converts capital + weights + prices into share counts (integer or fractional), accounting for commission schedules.

**`PortfolioRebalancer`:** detects weight drift against targets; generates buy/sell trade list with share counts.

**`DCAScheduler`:** dollar-cost averaging schedule with configurable frequency and periods.

**`PerformanceAttributor`:** Brinson attribution (allocation effect + selection effect + interaction).

---

### Factor Analysis (`src/factors/`)

**`FamaFrenchAnalyzer`:** OLS regression against 3-factor (MKT-RF, SMB, HML) or 5-factor (adds RMW, CMA) models. Factor data from Ken French library via `pandas_datareader`; synthetic fallback if unavailable.

**`StyleFactorAnalyzer`:** computes 5 style scores per asset — Momentum (12-1 month), Value (inverse valuation), Quality (ROE stability), Low Volatility (inverse realized vol), Size (inverse log market cap).

**`SectorAttribution` + `BrinsonAttribution`:** sector weight/return contributions; allocation, selection, and interaction effects.

**`FactorRiskDecomposition`:** systematic vs. idiosyncratic variance split; factor stress scenarios.

---

### Simulation (`src/simulation/`)

**`MonteCarloSimulator`** supports 4 methods:

| Method | Description |
|---|---|
| `gbm` | Geometric Brownian Motion: dP = muPdt + sigmaPdW |
| `bootstrap` | Block bootstrap preserving autocorrelation |
| `student_t` | Fat-tailed t-distribution for better tail modeling |
| `jump_diffusion` | GBM with Poisson jump component |

Output: `ndarray (n_simulations, horizon_days+1)`. `analyze_results()` returns mean final value, probability of loss, percentiles, mean max drawdown.

**`StressTester`** with 7 pre-loaded historical scenarios:

| Scenario | Period | Equity Drop | Vol Spike |
|---|---|---|---|
| 2008 Financial Crisis | 2008-09 to 2009-03 | -55% | 4.0x |
| COVID-19 Crash | 2020-02 to 2020-03 | -34% | 5.0x |
| 2018 Volatility Spike | 2018-10 to 2018-12 | -20% | 3.0x |
| 2022 Bear Market | 2022-01 to 2022-10 | -25% | 2.5x |
| Dot-com Bubble | 2000-03 to 2002-10 | -49% | 3.5x |
| Gulf War | 1990-07 to 1990-10 | -20% | 2.0x |
| Asian Crisis | 1997-07 to 1997-12 | -15% | 2.5x |

---

### Report Generation (`src/reports/`)

`ReportGenerator` (reportlab) + 3 templates (Portfolio Summary, Performance Review, Risk Dashboard) + `ReportChartGenerator` (6 matplotlib/seaborn chart types: allocation pie, performance line, drawdown area, correlation heatmap, monthly returns heatmap, return distribution histogram).

---

### Standalone Valuation Pipeline (`stock_valuer.py`)

Three-stage institutional screener:

```
analyze_stocks(tickers, wacc=0.10)
    Stage 1: magic_formula_screen()
             Filter: market_cap >= $500M, positive EBIT, debt/EBITDA <= 3x
             Rank: EBIT/EV (earnings yield) + ROIC rank
    Stage 2: multi_factor_score()   [top 30% by combined rank]
             Score 0-100 across Quality (30), Value (25), Momentum (20),
             Growth (15), Health (10) — sector-normalised thresholds
    Stage 3: reverse_dcf()          [score >= 65]
             Solve for implied growth rate via scipy.brentq on (-0.50, 3.0)
             Verdict: Reasonable / Stretched / Extreme
```

Rating labels: >=80 Strong Buy, >=65 Buy, >=50 Hold, >=35 Underweight, <35 Avoid.

Key design: `_coalesce(*values)` preserves legitimate 0.0 values (unlike Python `or`); Sloan accruals are signed (not absolute); single `yf.Ticker()` instantiation per ticker via `_fetch_all()`.

---

### Key Formulas

| Metric | Formula |
|---|---|
| Portfolio variance | sigma^2_p = w'Sw |
| Marginal risk contribution | MRC_i = (Sw)_i |
| Risk contribution (%) | RC%_i = w_i * (Sw)_i / sigma^2_p ; sum = 1 |
| Diversification Ratio | DR = sum(w_i * sigma_i) / sigma_p |
| Beta to portfolio | beta_i = (Sw)_i / sigma^2_p |
| Component VaR (normal) | CVaR_i = z * RC_i / sigma_p ; sum = z * sigma_p |
| ENB (Effective Number of Bets) | exp(-sum(lambda_i * ln(lambda_i))) where lambda_i = PCA eigenvalue shares |
| Sharpe Ratio | (mu_ann - rf) / sigma_ann |
| Sortino Ratio | (mu_ann - rf) / downside_deviation_ann |
| Calmar Ratio | mu_ann / |MDD| |
| Sloan Accruals | (NI - OCF) / avg_total_assets (signed) |
| Reverse DCF PV | sum[t=1..10] FCF*(1+g)^t/(1+WACC)^t + terminal_value = Market_Cap |

---

### Data Flow

```
User input (tickers, dates, capital)
    |
    v
DataManager.get_returns()
    |-- DataCache hit --> return cached
    |-- miss --> YFinance / AV / TwelveData / FMP --> validate --> cache
    |
    v
PortfolioOptimizer(returns)
    |-- .optimize(method, constraints)
    |       --> scipy SLSQP --> weights
    |
    v
RiskMetrics(returns) + VaRCalculator(returns)
    |-- portfolio_returns = (returns * weights).sum(axis=1)
    |-- VaR, CVaR, drawdown, Sharpe, GARCH, ...
    |
    v
StressTester(returns, weights, capital)
    |-- historical scenarios --> date-slice --> cumulative returns
    |-- MonteCarloSimulator --> GBM/bootstrap/t/jump --> paths
    |
    v
FamaFrenchAnalyzer + StyleFactorAnalyzer + FactorRiskDecomposition
    |-- OLS regressions, style scores, decomposition
    |
    v
ReportGenerator + template --> PDF bytes --> st.download_button()
```

### Session State Contract

All Streamlit pages share state through `st.session_state`. The keys written by each page are:

**Page 1 writes:** `tickers`, `portfolio_data` (`{prices, returns, start_date, end_date, current_prices}`), `current_holdings`, `holdings_tracker`, `current_portfolio_weights`, `settings` (`{total_capital, optimization_method, risk_free_rate, max_weight, min_weight, allow_fractional, target_volatility}`)

**Page 2 writes:** `optimization_result` (`{weights, expected_return, volatility, sharpe_ratio, method}`), `optimizer`, `weights`, `returns`, `prices`, `portfolio_value`, `metrics`

**Page 4 writes:** `mc_results` (`{values, analysis}`)

**Page 7 writes:** `generated_report`, `report_filename`

Pages 3-7 read from `portfolio_data` + `weights` + `optimization_result`. If any upstream page has not run, they display a "load data first" warning.

---

## Web Application Pages

| Page | File | Description |
|---|---|---|
| Home | `Home.py` | Feature overview, quick start, session init |
| Portfolio Input | `1_Portfolio_Input.py` | Holdings entry (tickers + shares) or manual tickers; diversity analysis |
| Optimization | `2_Optimization.py` | Method selection, run optimizer, rebalancing diff, position sizing, efficient frontier, method comparison |
| Risk Analytics | `3_Risk_Analytics.py` | VaR, drawdown, correlation, volatility tabs; deep risk analysis (enhanced VaR, tail risk, Monte Carlo); hedging effectiveness (risk contribution, beta classification, diversification waterfall, ENB) |
| Stress Testing | `4_Stress_Testing.py` | Historical scenarios, Monte Carlo (4 methods), custom stress, hedge effectiveness during stress |
| Monitoring | `5_Monitoring.py` | Rebalancing drift, performance attribution, DCA scheduler |
| Factor Analysis | `6_Factor_Analysis.py` | Fama-French 3/5-factor, sector attribution, style factors, risk decomposition |
| Reports | `7_Reports.py` | PDF generation with configurable charts and templates |

---

## Known Issues & Action Items

### Active Bugs

| Severity | Location | Issue |
|---|---|---|
| Low | `3_Risk_Analytics.py:243` | EWMA Volatility panel plots `rolling(20).std()` instead of the computed `ewma_vol` series. Cosmetic mislabel only. |
| Low | `src/reports/charts.py` | Uses deprecated `pd.date_range(freq='M')` — pandas 2.x requires `'ME'`. Causes ValueError in monthly returns heatmap chart. |

### Action Items

**Critical:**
- Fix `freq='M'` → `'ME'` in `src/reports/charts.py` — find all occurrences with `grep -n "freq='M'" src/reports/charts.py`
- Ensure `reportlab` is installed before running the Reports page

**High priority:**
- Expose Black-Litterman in the UI — `black_litterman.py` is complete but `2_Optimization.py` has no option for it. Add `"Black-Litterman": "black_litterman"` to `METHOD_MAP`.
- Fix the EWMA mislabel — `3_Risk_Analytics.py:243`: replace `rolling(20).std()` with `ewma_vol` already on line 238.

**Medium priority:**
- Remove unused heavy dependencies: `cvxpy`, `PyPortfolioOpt`, `riskfolio-lib`, `empyrical`, `quantstats` are in `requirements.txt` but never imported — ~500MB of unnecessary install overhead.
- Consolidate the two HRP implementations: `src/optimization/hrp.py` duplicates code in `optimizers.py`.
- Add centralized session state validation so pages give consistent "please load data first" behavior.
- Add CLI entry point for `stock_valuer.py` with argparse.

**Low priority:**
- Plot the actual `ewma_vol` series (computed but discarded) alongside rolling vol.
- Color the efficient frontier curve by Sharpe ratio.
- Surface `data_quality_score` from `multi_factor_score()` in `print_report()` output.
