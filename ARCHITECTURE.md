# PortfolioOptimizer — Architecture & Status Report

> Generated: 2026-05-19  
> Branch: `claude/stock-portfolio-tracker-KUfie`  
> Total commits: 25 | Total lines of code: ~19,100 across 54 Python files

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Technology Stack](#3-technology-stack)
4. [Module Architecture](#4-module-architecture)
   - 4.1 Data Layer (`src/data/`)
   - 4.2 Optimization Engine (`src/optimization/`)
   - 4.3 Risk Analytics (`src/risk/`)
   - 4.4 Portfolio Management (`src/portfolio/`)
   - 4.5 Factor Analysis (`src/factors/`)
   - 4.6 Simulation & Scenarios (`src/simulation/`)
   - 4.7 Report Generation (`src/reports/`)
   - 4.8 Utilities (`src/utils/`)
5. [Streamlit Web Application](#5-streamlit-web-application)
6. [Standalone Valuation Pipeline](#6-standalone-valuation-pipeline-stock_valuerpy)
7. [Data Flow & Architecture Patterns](#7-data-flow--architecture-patterns)
8. [Session State Contract](#8-session-state-contract)
9. [Key Algorithms & Formulas](#9-key-algorithms--formulas)
10. [Development History (Git Log)](#10-development-history-git-log)
11. [Test Suite Status](#11-test-suite-status)
12. [Known Issues & Technical Debt](#12-known-issues--technical-debt)
13. [Action Items](#13-action-items)

---

## 1. Project Overview

**PortfolioOptimizer** is a full-stack quantitative finance application providing institutional-grade portfolio construction, risk analytics, stress testing, factor analysis, and PDF reporting through a Streamlit web interface. It is intended as a calculation-assistance tool, not investment advice.

**Core capabilities:**

| Capability | Description |
|---|---|
| Data Ingestion | Multi-source fallback (yfinance → Alpha Vantage → Twelve Data → FMP) with caching |
| Portfolio Optimization | 7 algorithms: Max Sharpe, Min Vol, Risk Parity, HRP, Max Diversification, Max Return, Equal Weight |
| Risk Analytics | VaR/CVaR (3 methods), GARCH, drawdown, Sharpe/Sortino/Calmar/Omega, correlation |
| Deep Risk Analysis | Enhanced VaR comparison, tail risk (JB test, QQ plot), Monte Carlo, component VaR, ENB via PCA |
| Hedging Effectiveness | Beta classification, risk contribution decomposition, diversification waterfall |
| Stress Testing | 7 historical crises + parametric + Monte Carlo (4 methods), hedge analysis per scenario |
| Factor Analysis | Fama-French 3/5-factor, style factors, Brinson attribution, risk decomposition |
| Portfolio Tracking | Holdings input (tickers + shares), diversity metrics (HHI, Gini, effective stocks) |
| Rebalancing | Drift detection, trade recommendations, DCA scheduler, performance attribution |
| Reporting | PDF generation with 6 chart types and 3 report templates |
| Stock Valuation | Standalone 3-stage pipeline: Magic Formula → Multi-Factor Score → Reverse DCF |

---

## 2. Repository Layout

```
PortfolioOptimizer/
│
├── app/                                 # Streamlit multi-page application
│   ├── Home.py                          # Landing page, session state init        (162 lines)
│   └── pages/
│       ├── 1_Portfolio_Input.py         # Holdings entry & data loading            (494 lines)
│       ├── 2_Optimization.py            # Optimization + rebalancing UI            (396 lines)
│       ├── 3_Risk_Analytics.py          # Risk dashboard + deep analysis           (674 lines)
│       ├── 4_Stress_Testing.py          # Historical scenarios + Monte Carlo       (371 lines)
│       ├── 5_Monitoring.py              # Rebalancing + attribution + DCA          (324 lines)
│       ├── 6_Factor_Analysis.py         # FF factors + style + decomposition       (579 lines)
│       └── 7_Reports.py                 # PDF report generation UI                 (358 lines)
│
├── src/                                 # Core library
│   ├── data/
│   │   ├── data_manager.py              # Orchestrated multi-source fetcher        (448 lines)
│   │   ├── sources.py                   # YFinance/AV/TwelveData/FMP adapters     (521 lines)
│   │   ├── cache.py                     # TTL-based disk cache                    
│   │   └── validators.py                # OHLCV data validation utilities          
│   │
│   ├── optimization/
│   │   ├── optimizers.py                # PortfolioOptimizer (7 methods)           (518 lines)
│   │   ├── constraints.py               # PortfolioConstraints dataclass           
│   │   ├── black_litterman.py           # Black-Litterman model                    (432 lines)
│   │   └── hrp.py                       # Standalone HRP implementation            
│   │
│   ├── risk/
│   │   ├── metrics.py                   # RiskMetrics (20+ metrics)                (673 lines)
│   │   ├── var.py                       # VaRCalculator (historical/param/MC)      (559 lines)
│   │   └── garch.py                     # GARCHModel + ewma_volatility             (495 lines)
│   │
│   ├── portfolio/
│   │   ├── calculator.py                # PositionCalculator (share sizing)        (430 lines)
│   │   ├── holdings.py                  # HoldingsTracker + diversity metrics      
│   │   └── rebalancer.py                # Rebalancer + DCAScheduler + Attribution  (489 lines)
│   │
│   ├── factors/
│   │   ├── fama_french.py               # FamaFrenchAnalyzer (3/5-factor)         (436 lines)
│   │   ├── attribution.py               # Brinson attribution + sector analysis    (445 lines)
│   │   ├── style.py                     # StyleFactorAnalyzer                      (474 lines)
│   │   └── decomposition.py             # FactorRiskDecomposition                  (399 lines)
│   │
│   ├── simulation/
│   │   ├── monte_carlo.py               # MonteCarloSimulator (4 methods)          (547 lines)
│   │   └── scenarios.py                 # StressTester + HISTORICAL_SCENARIOS      (527 lines)
│   │
│   ├── reports/
│   │   ├── generator.py                 # ReportGenerator (reportlab PDF)          
│   │   ├── templates.py                 # 3 report templates                       (513 lines)
│   │   └── charts.py                    # ReportChartGenerator (6 chart types)     (501 lines)
│   │
│   └── utils/
│       ├── helpers.py                   # validate_tickers, format_*, calc helpers 
│       └── logger.py                    # loguru-based structured logging           
│
├── tests/                               # Unit test suite (256 tests total)
│   ├── test_data.py
│   ├── test_optimization.py
│   ├── test_risk.py
│   ├── test_portfolio.py
│   ├── test_factors.py
│   ├── test_simulation.py
│   └── test_reports.py
│
├── stock_valuer.py                      # Standalone 3-stage valuation pipeline    (1,295 lines)
├── test_holdings.py                     # Ad-hoc holdings tracker tests
├── test_holdings_simple.py              # Simplified holdings tests
├── test_indonesian_stocks.py            # Regional market coverage tests
├── requirements.txt                     # 50 dependencies
├── setup.py                             # Package config (v1.0.0)
├── .env.example                         # API key template
└── README.md                            # Basic project documentation
```

**Line count by category:**

| Category | Files | Lines |
|---|---|---|
| Streamlit Pages | 8 | 3,358 |
| src/risk | 3 | 1,727 |
| src/simulation | 2 | 1,074 |
| src/factors | 4 | 1,754 |
| src/reports | 3 | 1,014+ |
| src/optimization | 4 | 950+ |
| src/data | 4 | 969+ |
| src/portfolio | 3 | 919+ |
| stock_valuer.py | 1 | 1,295 |
| tests | 7 | ~2,000 |
| **Total** | **54** | **~19,100** |

---

## 3. Technology Stack

### Core Compute
| Package | Version | Role |
|---|---|---|
| pandas | ≥2.0.0 | DataFrames, time-series operations |
| numpy | ≥1.24.0 | Array math, matrix operations |
| scipy | ≥1.10.0 | SLSQP optimizer, hierarchical clustering, stats tests |

### Financial Data
| Package | Version | Role |
|---|---|---|
| yfinance | ≥0.2.40 | Primary OHLCV + fundamentals (ticker.info, financials, cashflow) |
| alpha_vantage | ≥2.3.1 | Fallback price data (500 calls/day) |
| pandas_datareader | ≥0.10.0 | Fama-French factor data from FRED/Ken French |
| requests | ≥2.31.0 | Twelve Data + FMP REST APIs |

### Optimization & Risk
| Package | Version | Role |
|---|---|---|
| cvxpy | ≥1.4.0 | Convex optimization (available but not currently used in main optimizer) |
| PyPortfolioOpt | ≥1.5.5 | Portfolio optimization library (available, not currently used) |
| arch | ≥6.3.0 | GARCH(1,1) volatility modeling |
| scikit-learn | ≥1.3.0 | PCA for ENB calculation |
| statsmodels | ≥0.14.0 | OLS for Fama-French regressions |

### Visualization & UI
| Package | Version | Role |
|---|---|---|
| streamlit | ≥1.31.0 | Web interface framework |
| plotly | ≥5.18.0 | All interactive charts |
| matplotlib | ≥3.7.0 | Static charts for PDF reports |
| seaborn | ≥0.13.0 | Statistical chart styling |
| reportlab | ≥4.0.0 | PDF generation |

### Utilities
| Package | Version | Role |
|---|---|---|
| loguru | ≥0.7.0 | Structured logging with rotation |
| tenacity | ≥8.2.3 | Exponential-backoff retry for network calls |
| python-dotenv | ≥1.0.0 | API key management from `.env` |
| rich | ≥13.0.0 | Terminal formatting for stock_valuer.py report |
| tabulate | ≥0.9.0 | Table formatting fallback |
| joblib | ≥1.3.0 | Parallel computation utilities |

---

## 4. Module Architecture

### 4.1 Data Layer (`src/data/`)

**`data_manager.py` — `DataManager` class**

Central orchestrator for all market data. Implements a priority-ordered fallback chain across 4 data sources with TTL caching and transparent failover.

```
get_price_data(tickers, start_date, end_date)
    → Check DataCache (TTL: 86400s historical, 3600s intraday)
    → Try YFinanceSource    (primary, batched in groups of 5)
    → Try AlphaVantageSource (500 req/day, 12s rate limit)
    → Try TwelveDataSource   (800 req/day, 8s rate limit)
    → Try FMPSource          (0.5s between calls)
    → Validate with DataValidator
    → Cache result
    → Return prices DataFrame (dates × tickers)
```

Key methods:
- `get_price_data(tickers, start, end, validate, use_cache)` → `pd.DataFrame`
- `get_returns(tickers, start, end, method="simple")` → `pd.DataFrame`
- `get_ohlcv_data(tickers, start, end)` → `dict`
- `get_current_prices(tickers)` → `pd.Series`
- `get_sector_info(tickers)` → `pd.DataFrame`
- `clear_cache()`, `get_cache_stats()`

**`sources.py` — Individual Source Adapters**

All inherit from `BaseDataSource` with `is_available` property and `fetch_prices()` method. Sources mark themselves unavailable on repeated failures to avoid unnecessary retries.

- **YFinanceSource**: Uses `yf.download()` with batch splitting; handles single vs multi-ticker differences in output structure; 3 retries with random jitter (0.1–0.5s)
- **AlphaVantageSource**: `TIME_SERIES_DAILY_ADJUSTED` endpoint; raises `RateLimitError` on HTTP 429
- **TwelveDataSource**: REST call with date range; parses `datetime` + `close` columns
- **FMPSource**: `historical-price-full` endpoint; uses `adjClose` or falls back to `close`

---

### 4.2 Optimization Engine (`src/optimization/`)

**`optimizers.py` — `PortfolioOptimizer` class**

All numeric optimization uses `scipy.optimize.minimize` with the SLSQP method. The constructor pre-computes annualized mean returns and covariance matrix once.

```python
PortfolioOptimizer(returns: pd.DataFrame, risk_free_rate=0.02, frequency=252)
```

| Method key | Algorithm | Objective |
|---|---|---|
| `max_sharpe` | SLSQP | Maximize `(μ - rf) / σ` |
| `min_volatility` | SLSQP | Minimize `√(w'Σw)` |
| `max_return` | SLSQP | Maximize `w·μ` |
| `max_diversification` | SLSQP | Maximize `(w·σ_i) / √(w'Σw)` |
| `risk_parity` | SLSQP | Minimize `Σ(RC_i − RC_target)²` |
| `hrp` | Hierarchical clustering | Recursive bisection, inverse-variance weights |
| `equal_weight` | Analytical | `1/n` for all assets |

**HRP implementation detail** (`_optimize_hrp` + `_hrp_recursive_bisection`):
1. Compute correlation matrix → distance `d_ij = √(0.5(1−ρ_ij))`
2. Single-linkage hierarchical clustering on condensed distance matrix
3. `scipy.cluster.hierarchy.leaves_list()` for quasi-diagonal ordering
4. Recursive bisection: split cluster in half, allocate proportional to inverse cluster variance
5. Cluster variance uses inverse-variance weighted sub-portfolio

**Efficient Frontier** (`efficient_frontier`):
- Sweeps target returns from min-vol to max-return result
- Per point: SLSQP with two equality constraints (`Σw=1`, `w·μ=target`)
- Options: `ftol=1e-9, maxiter=1000` (critical — was the root cause of prior broken frontier bug)
- Accepts near-converged solutions: `result.success OR (result.fun > 0 AND |Σw−1| < 1e-3)`
- Sorts output by volatility before returning

**`constraints.py` — `PortfolioConstraints`**

```python
PortfolioConstraints(
    min_weight=0.0, max_weight=1.0,
    min_position_size=0.0, max_position_size=1.0,
    sector_limits: Dict[str, float] = None,
    max_turnover: float = None,
    target_volatility: float = None,
    long_only: bool = True
)
```

Methods: `get_bounds()`, `get_sector_constraints()`, `get_turnover_constraint()`, `apply_minimum_position()`, `check_constraints()`

**`black_litterman.py`** (432 lines) — Full Black-Litterman model implementation with prior (market equilibrium) + views (P, Q matrices) → posterior expected returns. Available but not exposed in the current UI.

---

### 4.3 Risk Analytics (`src/risk/`)

**`metrics.py` — `RiskMetrics` class** (673 lines)

```python
RiskMetrics(returns: pd.DataFrame, risk_free_rate=0.02, frequency=252, benchmark_returns=None)
```

**Volatility family:**
- `historical_volatility(window, annualize)` → rolling or full-period
- `parkinson_volatility(high, low, window, annualize)` → range-based (more efficient)
- `downside_deviation(threshold=0, annualize)` → semi-deviation below threshold

**Performance ratios:**
- `sharpe_ratio()` → `(μ × freq) / (σ × √freq)`
- `sortino_ratio(threshold=0)` → uses downside deviation denominator
- `calmar_ratio(prices)` → annual return / max drawdown
- `omega_ratio()` → sum of gains / |sum of losses|

**Drawdown family:**
- `max_drawdown(prices)` → peak-to-trough maximum
- `average_drawdown(prices)` → mean of all drawdown periods
- `ulcer_index(prices)` → RMS of drawdown series (risk-of-regret measure)
- `drawdown_series(prices)` → full time series

**Summary:**
- `summary_table(prices)` → comprehensive DataFrame (return, vol, Sharpe, Sortino, MDD, Calmar, skew, kurtosis)

**`var.py` — `VaRCalculator` class** (559 lines)

- `historical_var(confidence)` → quantile of empirical return distribution
- `parametric_var(confidence)` → normal distribution assumption
- `monte_carlo_var(n_simulations, confidence)` → simulation-based
- `calculate_all(confidence)` → all methods in one DataFrame

**`garch.py` — `GARCHModel` class** (495 lines)

GARCH(1,1) volatility forecasting using the `arch` library. Also provides `ewma_volatility(returns, decay=0.94)` for exponentially weighted moving average volatility — note: the Streamlit UI labels the EWMA panel but currently plots rolling 20-day std instead (pre-existing cosmetic bug).

---

### 4.4 Portfolio Management (`src/portfolio/`)

**`holdings.py` — `HoldingsTracker`**

```python
HoldingsTracker(holdings: Dict[str, float], current_prices: Optional[pd.Series])
# holdings = {"AAPL": 10.0, "MSFT": 5.0, ...}
```

Diversity metrics computed:
- `herfindahl_index` (HHI = Σwᵢ²): concentration measure, 0→1
- `effective_stocks` (1/HHI): equivalent equal-weight count
- `top_3_concentration`, `top_5_concentration`
- `diversification_ratio`: (Σwᵢσᵢ) / σ_portfolio
- `gini_coefficient`: 0 = perfect equality, 1 = full concentration

Ratings: Excellent / Good / Moderate / Poor + actionable text recommendations.

**`calculator.py` — `PositionCalculator`**

```python
PositionCalculator(
    total_capital, weights, current_prices,
    allow_fractional=False,
    commission_per_trade=0.0, commission_per_share=0.0, min_commission=0.0
)
```

- `calculate_positions()` → DataFrame: Weight, Price, Shares, Actual Amount, Remainder
- `get_summary()` → `{total_invested, unallocated_cash, unallocated_pct}`
- Handles integer rounding, commission deduction, and fractional share toggling

**`rebalancer.py`** (489 lines) — Three classes:

1. **`PortfolioRebalancer`**: Detects drift from target weights; generates buy/sell trade list with share counts and dollar amounts; calculates turnover percentage
2. **`PerformanceAttributor`**: Brinson attribution (allocation effect + selection effect + interaction); return/risk contribution breakdown per asset
3. **`DCAScheduler`**: Dollar-cost averaging schedule; configurable frequency (weekly/monthly/quarterly) and number of periods; output includes per-period amounts and per-asset allocations

---

### 4.5 Factor Analysis (`src/factors/`)

**`fama_french.py` — `FamaFrenchAnalyzer`** (436 lines)

```python
FamaFrenchAnalyzer(returns: pd.DataFrame, factor_data: Optional[pd.DataFrame], risk_free_rate=0.02)
```

- Factor data sourced from `pandas_datareader` (Ken French data library via FRED). Falls back to synthetic data if unavailable.
- `analyze_asset(asset, model='3'|'5')` → `{betas, alpha, p_values, t_stats, r_squared}` via OLS (`statsmodels`)
- `analyze_portfolio(weights, model)` → portfolio-level factor exposures
- `factor_contribution(weights, model)` → return attribution by factor
- **3-Factor**: MKT-RF, SMB (size), HML (value)
- **5-Factor**: adds RMW (profitability), CMA (investment)

**`attribution.py`** (445 lines) — **`SectorAttribution`** + **`BrinsonAttribution`**

Brinson decomposition:
- Allocation effect: overweight high-performing sectors
- Selection effect: picking winners within sectors
- Interaction effect: combined skill

**`style.py` — `StyleFactorAnalyzer`** (474 lines)

Computes asset-level scores for 5 style factors:
- **Momentum**: 12−1 month price return
- **Value**: inverse of P/B or EV/EBITDA proxy
- **Quality**: ROE stability, earnings quality
- **Low Volatility**: inverse realized volatility
- **Size**: inverse log market cap

**`decomposition.py` — `FactorRiskDecomposition`** (399 lines)

- Systematic vs. idiosyncratic variance split
- Factor beta estimation per asset
- Risk contribution by factor
- Stress scenario simulation by factor shock

---

### 4.6 Simulation & Scenarios (`src/simulation/`)

**`monte_carlo.py` — `MonteCarloSimulator`** (547 lines)

```python
MonteCarloSimulator(returns: pd.DataFrame, weights=None, initial_value=10000, frequency=252)
```

| Method | Description |
|---|---|
| `simulate_gbm` | Geometric Brownian Motion: `dP = μPdt + σPdW` |
| `simulate_bootstrap` | Block bootstrap preserving autocorrelation structure |
| `simulate_student_t` | Fat-tailed t-distribution for better tail modeling |
| `simulate_jump_diffusion` | GBM plus Poisson jump component |

Output: `np.ndarray` shape `(n_simulations, horizon_days+1)`

`analyze_results(sim_values)` → `{mean_final_value, prob_loss, percentiles: {5th…95th}, mean_max_drawdown}`

**`scenarios.py` — `StressTester` + `HISTORICAL_SCENARIOS`** (527 lines)

Pre-loaded scenarios (each with `start_date`, `end_date`, `description`, `characteristics`):

| Key | Event | Equity Drop | Vol Spike |
|---|---|---|---|
| `gfc_2008` | 2008 Financial Crisis | −55% | 4.0× |
| `covid_2020` | COVID-19 Crash | −34% | 5.0× |
| `vol_spike_2018` | 2018 Volatility Spike | −20% | 3.0× |
| `bear_2022` | 2022 Bear Market | −25% | 2.5× |
| `dotcom_2000` | Dot-com Bubble | −49% | 3.5× |
| `gulf_war_1990` | Gulf War | −20% | 2.0× |
| `asian_crisis_1997` | Asian Crisis | −15% | 2.5× |

Methods:
- `run_all_historical()` → DataFrame of `{Scenario, Portfolio Return, Portfolio Loss ($)}`
- `parametric_stress(equity_shock, vol_mult, corr_adj)` → stressed portfolio metrics
- `sensitivity_analysis()` → DataFrame sweeping shock levels −50% to +10%

---

### 4.7 Report Generation (`src/reports/`)

**`generator.py` — `ReportGenerator`**

```python
ReportGenerator(title="Portfolio Analysis Report", author="System", page_size="letter")
```

Uses `reportlab` for PDF assembly. Custom paragraph styles, color palette (`#1a1a2e`, `#16213e`). Methods: `add_title`, `add_section`, `add_paragraph`, `add_table`, `add_chart`, `add_metrics_row`, `add_spacer`, `add_page_break`, `generate_bytes()`.

**`templates.py`** (513 lines) — Three templates:
- `PortfolioSummaryTemplate`: executive summary, holdings, metrics
- `PerformanceReviewTemplate`: period performance, asset contributions, monthly returns
- `RiskDashboardTemplate`: VaR/CVaR analysis, correlation matrix, stress test results

**`charts.py` — `ReportChartGenerator`** (501 lines)

All charts return PNG bytes for embedding in PDF:
- `allocation_pie_chart(weights)` — matplotlib pie
- `performance_line_chart(returns)` — cumulative return
- `drawdown_chart(returns)` — drawdown area fill
- `correlation_heatmap(returns)` — seaborn heatmap
- `monthly_returns_heatmap(returns)` — calendar heatmap
- `distribution_histogram(returns)` — return distribution with KDE

---

### 4.8 Utilities (`src/utils/`)

**`helpers.py`**
- `validate_tickers(input)` → `List[str]` — parses comma/semicolon/space-separated input, validates A-Z0-9.-+ format, max 10 chars
- `format_currency(value, currency, decimals)`, `format_percentage(value, decimals)`, `format_number(value, decimals)`
- `annualize_returns(daily_returns, frequency)`, `annualize_volatility(daily_vol, frequency)`
- `calculate_returns(prices, method="simple"|"log")` → DataFrame
- `chunk_list(lst, size)`, `parse_date(date_input)`

**`logger.py`**
- `get_logger(name)` → loguru `Logger` with format, level, rotation
- Log file: `logs/portfolio_optimizer.log`
- Console: colored output with timestamps

---

## 5. Streamlit Web Application

All 7 pages share state via `st.session_state`. Navigation flows naturally left-to-right: load data → optimize → analyze risk → stress test → monitor → factors → export.

### Page 1 — Portfolio Input (`1_Portfolio_Input.py`, 494 lines)

**Two input modes:**

**Mode A — Current Holdings** (ticker + shares):
- Fetches live prices via `DataManager`
- Computes portfolio value, weights
- Shows diversity dashboard (HHI, effective stocks, Gini, top-5 concentration)
- Renders `HoldingsTracker.get_diversity_recommendations()`

**Mode B — Manual Tickers**:
- Direct ticker entry or preset portfolios (Tech Giants, Diversified ETFs, Blue Chips)

**Shared configuration:**
- Lookback period: 1 / 2 / 3 / 5 years or custom date range
- Capital, risk-free rate, max/min weight constraints
- Fractional shares toggle, target volatility

**Session state written:** `tickers`, `portfolio_data`, `current_holdings`, `current_portfolio_weights`, `holdings_tracker`, `settings`

---

### Page 2 — Optimization (`2_Optimization.py`, 396 lines)

- Dropdown selects one of 6 methods (Max Sharpe, Min Vol, Risk Parity, HRP, Max Diversification, Equal Weight)
- "Run Optimization" → `PortfolioOptimizer.optimize()` → stores result
- **Rebalancing block** (shown only if holdings exist): current vs. target weight grouped bar chart, trade table (current shares → target shares → delta), buy/sell action lists
- **Position sizing table**: Weight, Price, Shares, Actual Amount, Remainder
- **Efficient Frontier** (on-demand button): scatter of individual assets + frontier curve + optimal portfolio star
- **Method Comparison** (on-demand button): table comparing return/vol/Sharpe across all 7 methods

---

### Page 3 — Risk Analytics (`3_Risk_Analytics.py`, 674 lines)

The largest UI page. Three logical sections:

**Section A — Existing 4-tab block:**
1. **VaR Analysis**: `VaRCalculator.calculate_all()` + distribution histogram
2. **Drawdowns**: drawdown area chart + metrics table (MDD, avg DD, ulcer index, Calmar)
3. **Correlations**: heatmap + highest-correlation pairs table
4. **Volatility**: rolling window slider + EWMA portfolio volatility

**Section B — Deep Risk Analysis (Module 2, added this session):**
- **Enhanced VaR**: historical + Normal VaR + CVaR at 90/95/99%; divergence metric; bar comparison
- **Tail Risk**: skewness, kurtosis, Sortino, omega ratio, Jarque–Bera normality test, per-asset Sortino bar, QQ plot, tail event percentile table
- **Monte Carlo**: `@st.cache_data`-cached Gaussian simulation, simulated VaR/CVaR table, 100-day portfolio path fan chart

**Section C — Hedging Effectiveness (Module 3, added this session):**
- **Risk Contribution**: weight vs. RC grouped bar, DR metric, HHI metrics, risk budget pie
- **Hedge Classification**: beta-based equity vs. hedge/diversifier label, beta bar chart, component VaR (normal, 95%)
- **Diversification Benefit**: undiversified vs. portfolio variance waterfall, pairwise variance heatmap
- **Effective Bets**: PCA-based ENB = exp(Shannon entropy of eigenvalue ratios), scree plot with dual y-axis, factor loadings heatmap

**Section D — Performance Ratios Summary** + CSV export

---

### Page 4 — Stress Testing (`4_Stress_Testing.py`, 371 lines)

**Tab 1 — Historical Scenarios:**
- "Run All" → `StressTester.run_all_historical()` → results table + colored bar chart
- Individual scenario picker → description, equity drop, vol spike

**Tab 2 — Monte Carlo Simulation:**
- Config: n_simulations (1K–50K), horizon (21–504 days), method (GBM/bootstrap/student_t/jump_diffusion)
- Results: mean final value, prob of loss, VaR 95%, mean max DD
- Distribution histogram + percentile lines + 100 sample paths with 5/50/95th percentile fan

**Tab 3 — Custom Stress:**
- Sliders: equity shock (-50→0%), vol multiplier (1–5×), correlation adjustment (0.5–1.0)
- `StressTester.parametric_stress()` → portfolio return, loss amount, ending value
- Sensitivity sweep: `stress_tester.sensitivity_analysis()` → curve of ending value vs. shock level

**Section: Hedging Effectiveness During Stress (added this session):**
- Button-triggered: classifies assets as equity-like (β ≥ 0.5) vs. hedge by full-period portfolio beta
- For each historical scenario: asset return bar chart (crimson/steelblue) + "★ Best Hedge" annotation on highest-returning hedge asset
- Right panel: hedge effectiveness score = `(hc_ret × hc_weight) / |portfolio_loss|` per hedge asset; verdict: Strong / Moderate / Weak

---

### Page 5 — Monitoring (`5_Monitoring.py`, 324 lines)

**Tab 1 — Rebalancing:**
- Simulates portfolio drift using recent returns
- Threshold slider (1–15%) triggers rebalance alert
- `PortfolioRebalancer.calculate_trades()` → buy/sell table with share counts
- Summary: total buys, total sells, turnover %

**Tab 2 — Performance Attribution:**
- Return contribution by asset (bar chart)
- Risk contribution by asset (pie chart)
- Detailed table: Weight, Return, Contribution %, Volatility

**Tab 3 — DCA Scheduler:**
- Config: total amount, number of periods, frequency (weekly/monthly/quarterly)
- `DCAScheduler` → schedule table with per-period amounts and per-asset allocations
- CSV download

Cumulative growth chart + total/annualized return/volatility summary panel at bottom.

---

### Page 6 — Factor Analysis (`6_Factor_Analysis.py`, 579 lines)

**Tab 1 — Fama-French Analysis:**
- 3-factor or 5-factor model selector
- Bar chart of portfolio factor betas
- Metrics: alpha (annualized), R², number of observations
- Factor return contribution pie chart
- Optional per-asset breakdown

**Tab 2 — Sector Attribution:**
- Sector weights pie chart
- Sector return contribution bar chart
- Brinson attribution table (allocation, selection, interaction effects)
- Sector correlation matrix heatmap

**Tab 3 — Style Factors:**
- 5-factor exposure bar chart: Momentum, Value, Quality, Low Volatility, Size
- Portfolio vs. equal-weight comparison
- Asset factor scores table
- Factor return attribution bar chart
- Top stocks by selected factor

**Tab 4 — Risk Decomposition:**
- Systematic vs. specific volatility metrics
- Risk pie chart
- Portfolio factor betas table
- Factor risk contributions table
- Factor stress scenarios (shock × beta → expected impact)
- Factor correlation matrix

---

### Page 7 — Reports (`7_Reports.py`, 358 lines)

- 4 report types: Portfolio Summary / Performance Review / Risk Dashboard / Complete Analysis
- Configuration: title, author, portfolio value, page size (Letter/A4)
- 6 chart checkboxes (allocation pie, performance, drawdown, correlation, monthly returns, distribution)
- Single "Generate Report" button → calls selected template → `ReportGenerator.generate_bytes()` → PDF download
- Displays: file size, generation time, section summary

---

## 6. Standalone Valuation Pipeline (`stock_valuer.py`)

A self-contained institutional-grade stock screener (1,295 lines) that runs independently of the Streamlit app.

### Architecture

```
analyze_stocks(tickers, wacc=0.10)
    ↓
Stage 1: magic_formula_screen(tickers)
    → Filter: market_cap ≥ $500M, sector not in EXCLUDED_SECTORS
    → Filter: positive EBIT, debt/EBITDA ≤ 3×
    → Rank: EBIT/EV (earnings yield) + ROIC rank
    → Output: DataFrame with combined_rank
    ↓
Stage 2: multi_factor_score(ticker, wacc)  [top 30% by rank]
    → 5 factor groups → 0–100 composite score
    → Output: Dict with score_breakdown, data_quality_score
    ↓
Stage 3: reverse_dcf(ticker, wacc)  [score ≥ 65]
    → Solve for implied growth rate g using scipy.optimize.brentq
    → Compare to historical FCF CAGR
    → Output: {implied_growth, verdict: Reasonable/Stretched/Extreme}
    ↓
print_report(df)
    → rich → tabulate → plain pandas fallback
```

### Multi-Factor Scoring (0–100)

| Group | Weight | Factors |
|---|---|---|
| Quality | 30 pts | ROIC, gross margin trend, FCF/NI ratio |
| Value | 25 pts | EV/EBIT, FCF yield, EV/EBITDA vs. sector median |
| Momentum | 20 pts | Price vs. 200-SMA, 12−1 month return |
| Growth | 15 pts | Revenue CAGR, EPS leverage, Sloan accruals ratio |
| Health | 10 pts | Interest coverage, Debt/EBITDA |

Scores are **sector-normalized** via `SECTOR_THRESHOLDS` dict covering Tech, Healthcare, Consumer Discretionary, Consumer Staples, and Industrials.

### Key Design Decisions

- `_coalesce(*values)` — returns first **non-None** value (preserves legitimate `0.0`, unlike Python `or`)
- `_fetch_all(ticker)` — single `yf.Ticker()` instantiation per ticker, returns all needed data objects
- Sloan accruals are **signed** (not absolute): `(NI − OCF) / avg_assets`; negative = healthy, positive = quality warning
- Reverse DCF uses `scipy.optimize.brentq` on bounds `(−0.50, 3.0)` for implied growth rate
- Rating labels: ≥80 Strong Buy, ≥65 Buy, ≥50 Hold, ≥35 Underweight, <35 Avoid

---

## 7. Data Flow & Architecture Patterns

### Master Data Flow

```
User (browser)
    │
    ▼
Streamlit Page (session_state read)
    │
    ├─► DataManager.get_returns() ──► DataCache ──► Source chain
    │                                                 (YFinance → AV → 12D → FMP)
    ▼
PortfolioOptimizer(returns)
    │
    ├─► .optimize(method, constraints) ──► scipy SLSQP ──► weights
    │
    ▼
RiskMetrics(returns) + VaRCalculator(returns)
    │
    ├─► portfolio_returns = (returns × weights).sum()
    ├─► VaR, CVaR, drawdown, Sharpe, Sortino, GARCH forecast
    │
    ▼
StressTester(returns, weights, capital)
    │
    ├─► run_all_historical() ──► date-slice scenarios ──► cumulative returns
    ├─► MonteCarloSimulator ──► GBM/bootstrap/t/jump ──► path simulation
    │
    ▼
FamaFrenchAnalyzer / StyleFactorAnalyzer / FactorRiskDecomposition
    │
    ├─► OLS regression on FF factors
    ├─► Style score computation
    │
    ▼
ReportGenerator + template
    │
    └─► PDF bytes ──► st.download_button()
```

### Design Patterns Used

| Pattern | Where |
|---|---|
| Strategy | Optimization method selection (`method_map` dict) |
| Adapter | Data source wrappers (all expose same `fetch_prices` interface) |
| Template Method | Report templates (`build()` + shared `ReportGenerator`) |
| Decorator | `tenacity.retry` on network calls; `@st.cache_data` on expensive computations |
| Facade | `DataManager` hides 4 sources behind a single API |
| Session Object | Streamlit `session_state` as shared state bus between pages |

---

## 8. Session State Contract

All pages communicate via these `st.session_state` keys:

```python
{
    # Set by Page 1
    'tickers': List[str],
    'portfolio_data': {
        'prices':          pd.DataFrame,   # dates × tickers
        'returns':         pd.DataFrame,   # dates × tickers (simple returns)
        'start_date':      datetime,
        'end_date':        datetime,
        'current_prices':  pd.Series       # ticker → latest price
    },
    'current_holdings':          Dict[str, float],   # {ticker: num_shares}
    'holdings_tracker':          HoldingsTracker,
    'current_portfolio_weights': pd.Series,          # ticker → weight (sum=1)
    'settings': {
        'total_capital':        float,
        'optimization_method':  str,
        'risk_free_rate':        float,
        'max_weight':            float,
        'min_weight':            float,
        'allow_fractional':      bool,
        'target_volatility':     Optional[float]
    },

    # Set by Page 2
    'optimization_result': {
        'weights':          pd.Series,
        'expected_return':  float,
        'volatility':       float,
        'sharpe_ratio':     float,
        'method':           str
    },
    'optimizer':          PortfolioOptimizer,    # reused for frontier/compare
    'weights':            pd.Series,
    'returns':            pd.DataFrame,
    'prices':             pd.DataFrame,
    'portfolio_value':    float,
    'metrics': {
        'annual_return':     float,
        'annual_volatility': float,
        'sharpe_ratio':      float
    },

    # Set by Page 4
    'mc_results': {
        'values':   np.ndarray,   # shape (n_sims, horizon)
        'analysis': Dict
    },

    # Set by Page 7
    'generated_report':  bytes,
    'report_filename':   str
}
```

**Important:** Pages 3–7 read `portfolio_data` + `returns` + `weights` + `optimization_result`. Any page that fails to set these keys will cause downstream pages to show their "load data first" warning.

---

## 9. Key Algorithms & Formulas

### Portfolio Math

| Metric | Formula |
|---|---|
| Portfolio variance | `σ²_p = w'Σw` |
| Marginal risk contribution | `MRC_i = (Σw)_i` |
| Risk contribution (var units) | `RC_i = w_i · (Σw)_i` |
| Risk contribution (%) | `RC%_i = RC_i / σ²_p` ; `Σ RC%_i = 1` |
| Diversification Ratio | `DR = (Σ w_i σ_i) / σ_p` |
| Beta to portfolio | `β_i = MRC_i / σ²_p = (Σw)_i / σ²_p` |
| Component VaR (normal) | `CVaR_i = z · RC_i / σ_p` ; `Σ CVaR_i = z · σ_p` |
| HRP distance | `d_ij = √(0.5(1 − ρ_ij))` |
| ENB (Effective Number of Bets) | `exp(−Σ λ_i · ln(λ_i))` where `λ_i` = PCA eigenvalue shares |
| Diversification Benefit | `DB = (Σ w_i σ_i)² − σ²_p` |

### Risk Metrics

| Metric | Formula |
|---|---|
| Sharpe Ratio | `(μ_ann − r_f) / σ_ann` |
| Sortino Ratio | `(μ_ann − r_f) / DD_ann` |
| Calmar Ratio | `μ_ann / |MDD|` |
| Omega Ratio | `Σ gains / Σ |losses|` (threshold = 0) |
| HHI (Herfindahl) | `Σ w_i²` |
| Effective Stocks | `1 / HHI` |
| Sloan Accruals | `(NI − OCF) / avg_total_assets` (signed, not absolute) |

### Reverse DCF

```
PV = Σ[t=1..10] FCF₀·(1+g)ᵗ / (1+WACC)ᵗ  +  FCF₀·(1+g)¹⁰·(1+TGR) / [(WACC−TGR)·(1+WACC)¹⁰]

Set PV = Market Cap, solve for g via brentq on (−0.50, 3.0)
TGR = 2.025%   (terminal growth rate)
```

---

## 10. Development History (Git Log)

| # | Commit | Description |
|---|---|---|
| 1 | `8fe0558` | Phase 1: Project setup and data infrastructure |
| 2 | `a7483b2` | Phase 2: Core optimization engine implementation |
| 3 | `37eb5d1` | Phase 1 & 2 test suite |
| 4 | `26823dd` | Phase 3: Risk analytics module |
| 5 | `87f44c0` | Phase 4: Stress testing and Monte Carlo |
| 6 | `f4b594d` | Phase 5: Position sizing and portfolio calculator |
| 7 | `a53330d` | Phase 6: Streamlit web interface |
| 8 | `362bd35` | Phase 7: Comprehensive test suite |
| 9 | `6ca50e6` | Phase 8: Factor analysis module |
| 10 | `6d94415` | Phase 9: Automated reporting module |
| 11 | `72e588d` | Tests for Phases 8 and 9 |
| 12 | `99167a4` | Fix session state and efficient frontier errors |
| 13 | `9c0734c` | Update README |
| 14 | `cbcc7ba` | Add holdings tracker with diversity analysis |
| 15 | `66faf8b` | PR #1 merge |
| 16 | `3dc3726` | Integrate holdings tracker with optimization and rebalancing |
| 17 | `0c21f9a` | PR #2 merge |
| 18 | `9bc7d0c` | Fix price fetching; enable all features for holdings-based portfolios |
| 19 | `73147ea` | Fix AttributeError with Streamlit date inputs |
| 20 | `b7a4094` | PR #3 merge |
| 21 | `135cc42` | **Add institutional-grade 3-stage stock valuation pipeline** |
| 22 | `b2bac70` | **Fix 5 bugs in stock_valuer.py fundamentals extraction** |
| 23 | `b09357d` | **Fix broken efficient frontier (3 bugs: maxiter, convergence, sort)** |
| 24 | `05c9e13` | **Add Module 2 (Deep Risk Analysis) and Module 3 (Hedging Effectiveness)** |
| 25 | `bb2bd7b` | **Fix 3 bugs: MC NaN propagation, best-hedge direction, offset sign error** |

Commits 21–25 are the work done in this session (current branch: `claude/stock-portfolio-tracker-KUfie`).

---

## 11. Test Suite Status

**Last run: 231 passed, 25 failed** (out of 256 total tests)

### Passing ✅

| Suite | Tests |
|---|---|
| `test_data.py` | All pass — data fetching, caching, validation, source fallback |
| `test_optimization.py` | All pass — 7 methods, constraints, efficient frontier, Black-Litterman |
| `test_portfolio.py` | All pass — holdings, position sizing, rebalancing, DCA, Brinson |
| `test_factors.py` | All pass — Fama-French 3/5-factor, style, decomposition, attribution |
| `test_simulation.py` | All pass — GBM, bootstrap, student_t, jump diffusion, all scenarios |
| `test_risk.py` (partial) | 34/36 pass — VaR, drawdown, Sharpe, Sortino, Calmar, correlation |

### Failing ❌

**`test_reports.py` — 23 failures (all same root cause)**
```
ImportError: reportlab is required for PDF generation.
```
`reportlab` is listed in `requirements.txt` but is not installed in the current test environment. All report tests fail at import time. Not a code bug — environment issue only.

**`test_risk.py` — 2 failures**
```python
# test_risk.py:364
assert np.float64(1.0) < 1   # GARCH persistence check: α+β should be < 1

# test_risk.py:379
assert inf < 1                # GARCH unconditional volatility overflow
```
The GARCH test assertions assume persistence (`α + β`) will be strictly less than 1.0, but the fitted GARCH(1,1) model is returning persistence = 1.0 (integrated GARCH / IGARCH behavior) with the synthetic test data, which causes unconditional variance to blow up to infinity. This is a test data / assertion design issue in the test, not necessarily a production bug.

---

## 12. Known Issues & Technical Debt

### Active Bugs

| Severity | File | Issue |
|---|---|---|
| Low | `3_Risk_Analytics.py:243` | Pre-existing: EWMA Volatility panel plots `rolling(20).std()` instead of using the computed `ewma_vol` Series. Cosmetic mislabel. |
| Low | `src/reports/charts.py` | Uses deprecated `pd.date_range(freq='M')` — pandas 2.x requires `'ME'` for month-end. Causes `ValueError` in monthly returns heatmap chart. |

### Technical Debt

| Area | Issue | Impact |
|---|---|---|
| `src/optimization/black_litterman.py` | Fully implemented (432 lines) but not exposed in the UI. No selectbox option in Page 2. | Feature gap |
| `src/optimization/hrp.py` | Standalone HRP file exists alongside the HRP implementation in `optimizers.py`. Duplication risk. | Maintainability |
| `cvxpy`, `PyPortfolioOpt`, `riskfolio-lib` | Listed as dependencies but not used anywhere in the codebase. Significant install overhead (~500MB). | Dependency bloat |
| `empyrical`, `quantstats` | Listed as dependencies but not imported anywhere in the codebase. | Dependency bloat |
| GARCH model tests | Test assertions fail on realistic synthetic data due to IGARCH edge case. Tests need better data fixtures or softer bounds. | Test reliability |
| `reportlab` not installed in CI | All 23 report tests fail in the container environment. | CI reliability |
| Session state fragility | Pages 3–7 silently degrade if upstream pages haven't run. No centralized state validation. | UX reliability |
| No `.env` / API key validation | App starts without checking if data source keys are set; first yfinance call will silently use free tier. | Config gap |

---

## 13. Action Items

### Critical (breaks features)

- [ ] **Fix `pd.date_range(freq='M')` → `'ME'`** in `src/reports/charts.py` — monthly returns heatmap crashes on pandas 2.x. Find all occurrences and update frequency strings.
- [ ] **Install `reportlab` in CI / test environment** — `pip install reportlab` in the runner or add it to a dev-requirements file. Currently blocks 23 tests.

### High Priority (significant gaps)

- [ ] **Expose Black-Litterman in the UI** — `black_litterman.py` is complete but Page 2 has no option for it. Add `"Black-Litterman": "black_litterman"` to `METHOD_MAP` and wire up views input form.
- [ ] **Fix EWMA volatility label** — `3_Risk_Analytics.py:243`: replace `rolling(20).std()` with the `ewma_vol` series already computed on line 238. One-line fix.
- [ ] **Fix GARCH test assertions** — `test_risk.py:364,379`: either use synthetic data that produces persistence < 1, or change the assertion to `<= 1.0` to allow IGARCH boundary, and guard the unconditional variance check for infinity.

### Medium Priority (quality / completeness)

- [ ] **Remove unused dependencies** — `cvxpy`, `PyPortfolioOpt`, `riskfolio-lib`, `empyrical`, `quantstats` are declared in `requirements.txt` but never imported. Remove them to reduce install time by ~500MB.
- [ ] **Consolidate HRP implementations** — `src/optimization/hrp.py` appears to duplicate what's in `optimizers.py`. Decide which is canonical and delete the other, or have `optimizers.py` import from `hrp.py`.
- [ ] **Add centralized session state validation** — Create a utility function (e.g. `require_data()`) that pages call at startup to give a consistent "please load data first" experience with a redirect button.
- [ ] **`stock_valuer.py` — CLI entry point** — Currently only usable as `python stock_valuer.py`. Expose as a proper CLI with argparse (`--tickers AAPL MSFT --wacc 0.10`) and add it to `setup.py` entry_points.
- [ ] **Environment validation at startup** — `Home.py` should check for `.env` / API key presence and display a configuration warning if none are set, explaining the fallback to yfinance free tier.

### Low Priority (polish)

- [ ] **Add Plotly chart for EWMA volatility** — Currently `ewma_volatility()` is called but the result is unused in the chart. Plot it alongside portfolio rolling vol for comparison.
- [ ] **Efficient Frontier: add Sharpe color coding** — The frontier is currently a plain blue line. Color points by Sharpe ratio (color scale) to visually identify the tangency portfolio.
- [ ] **Page 7 — Complete Analysis report type** — The "Complete Analysis" report type option exists in the UI dropdown but may not wire up all sections. Verify all 4 report types generate correctly once `reportlab` is installed.
- [ ] **Add `data_quality_score` to stock_valuer output display** — The `multi_factor_score()` function computes and returns a `data_quality_score` (0–1) per ticker indicating data completeness, but `print_report()` doesn't surface it in the table.
- [ ] **Write tests for new modules** — `3_Risk_Analytics.py` Modules 2 & 3 (Deep Risk Analysis, Hedging Effectiveness) and the stress-period hedging section in `4_Stress_Testing.py` have no corresponding unit tests. Add tests for the math: RC formula sumcheck, ENB bounds, hedge effectiveness sign.
- [ ] **Deprecate `test_holdings.py` / `test_holdings_simple.py`** — These ad-hoc root-level test scripts duplicate coverage that belongs in `tests/test_portfolio.py`. Migrate and delete.
