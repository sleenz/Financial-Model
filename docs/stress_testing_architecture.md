# Stress Testing Page — Architecture

## Overview

`app/pages/4_Stress_Testing.py` is a four-tab Streamlit page that exposes the full stress-testing engine. It reads portfolio data from Streamlit session state (populated by earlier pages) and routes each analysis request to the appropriate back-end module.

---

## Session State Contract

The page expects the following keys to be set by upstream pages before it loads:

| Key | Type | Source | Description |
|-----|------|--------|-------------|
| `returns` | `pd.DataFrame` | Page 2 (Data) | Daily log/pct returns; columns = tickers |
| `weights` | `np.ndarray` or `pd.Series` | Page 3 (Optimize) | Portfolio weights summing to 1 |
| `optimized_weights` | `pd.Series` | Page 3 (Optimize) | Optimizer output (highest priority) |
| `current_holdings` | `dict[str, float]` | Page 1 (Holdings) | ticker → shares |
| `current_prices` | `pd.Series` | Page 1 (Holdings) | ticker → latest price |
| `portfolio_value` | `float` | Page 1 (Holdings) | Total market value in USD |
| `tickers` | `list[str]` | Page 2 (Data) | Ordered list of asset tickers |

**Weight priority** (highest → lowest):

```
optimized_weights  →  current_holdings (value-weighted)  →  equal weight
```

---

## Tab Structure

```
┌─────────────────────────────────────────────────────────┐
│  4_Stress_Testing.py                                    │
│                                                         │
│  Tab 1: Historical Scenarios                            │
│  Tab 2: Monte Carlo Simulation                          │
│  Tab 3: Parametric / Sensitivity                        │
│  Tab 4: Sector Shock Analysis                           │
└─────────────────────────────────────────────────────────┘
```

---

## Tab 1 — Historical Scenarios

### Purpose
Replay portfolio performance through named historical crises using either actual per-stock returns or hardcoded characteristic shocks.

### Toggle Modes

| Mode | Backend Class | Return Source |
|------|---------------|---------------|
| **Actual returns** (default) | `HistoricalStressor` | yfinance daily prices for each stock over the crisis window |
| **Legacy (characteristics)** | `StressTester.run_all_historical()` | Hardcoded `equity_drop` from `HISTORICAL_SCENARIOS` dict |

### Actual Returns Flow (`HistoricalStressor`)

```
HistoricalStressorConfig
        │
        ▼
HistoricalStressor.run_all(tickers, weights, portfolio_value)
        │
        ├─ For each scenario in HISTORICAL_SCENARIOS (7 crises):
        │       ├─ yf.download(ticker, start, end)
        │       ├─ If ≥ min_data_points (5):  realized_return = cumulative pct change
        │       └─ Else: estimate beta from 252-day pre-crisis window
        │               beta clipped to [-3.0, 3.0]
        │               estimated_return = beta × index_return
        │
        ├─ Aggregates per-stock returns → weighted portfolio return
        └─ Returns dict[scenario_name → HistoricalScenarioResult]
```

**Scenarios included:**

| Scenario | Index | Period |
|----------|-------|--------|
| COVID-19 Crash | ^GSPC | 2020-02-19 → 2020-03-23 |
| 2008 Financial Crisis | ^GSPC | 2008-09-01 → 2009-03-31 |
| 1997 Asian Crisis | ^JKSE | 1997-07-01 → 1997-12-31 |
| 2013 Taper Tantrum | ^JKSE | 2013-05-22 → 2013-06-24 |
| 2022 Bear Market | ^GSPC | 2022-01-03 → 2022-10-12 |
| Dot-com Bubble | ^GSPC | 2000-03-10 → 2002-10-09 |
| 2018 Q4 Selloff | ^GSPC | 2018-10-01 → 2018-12-31 |

**Output displayed:** comparison table (portfolio return %, P&L $, ending value), expandable per-stock breakdown with `source` field (`"actual"` or `"beta_scaled"`).

---

## Tab 2 — Monte Carlo Simulation

### Purpose
Forward-looking simulation of portfolio value over a user-selected horizon.

### Backend: `MonteCarloSimulator`

```
src/simulation/monte_carlo.py
        │
        ├─ Method: GBM (Geometric Brownian Motion)
        ├─ Method: Bootstrap (historical block resampling)
        ├─ Method: Student-t (fat-tailed GBM)
        └─ Method: Jump-Diffusion (GBM + Poisson jumps)
```

### User Controls
- Simulation method selector
- Horizon (days)
- Number of paths
- Confidence level for VaR/CVaR display

### Output
- Fan-chart of simulated paths (Plotly)
- Distribution of terminal values
- VaR / CVaR at selected confidence
- Probability of loss exceeding threshold

---

## Tab 3 — Parametric / Sensitivity

### Purpose
Deterministic shock analysis across a range of equity drawdown magnitudes.

### Backend: `StressTester` (from `src/simulation/scenarios.py`)

```python
StressTester.parametric_stress(
    equity_shock,          # single uniform shock
    volatility_multiplier, # scales asset volatilities
    correlation_adjustment # sets all pairwise correlations
)

StressTester.sensitivity_analysis(
    shock_range=[-0.05, -0.10, ..., -0.50]
)
```

### Output
- Parametric stress result: portfolio return, P&L, ending value, base vs stressed volatility
- Sensitivity table: P&L at each shock level
- Bar chart of sensitivity results

### Additional: Reverse Stress Test
Given a target loss (e.g. -$50,000), computes the market shock required:

```
market_shock = target_loss / (portfolio_value × portfolio_beta)
```

---

## Tab 4 — Sector Shock Analysis

This is the most sophisticated tab, integrating five sub-models.

### Purpose
Apply sector-level shocks (e.g. "Tech Selloff: −25%") and propagate them through the full correlation structure, accounting for current market regime and tail dependence.

### Sub-Model Dependency Chain

```
                        ┌──────────────────────┐
                        │   SectorStressEngine │
                        │  (sector_stress.py)  │
                        └──────────┬───────────┘
                                   │ .fit(returns, sector_map)
                     ┌─────────────┼──────────────────┐
                     ▼             ▼                  ▼
           ┌─────────────┐  ┌───────────┐  ┌──────────────────┐
           │SectorBeta   │  │DCCGARCHModel│  │MarketRegimeDetect│
           │Analyzer     │  │(dcc_garch) │  │(regime_detection)│
           └─────────────┘  └─────┬─────┘  └────────┬─────────┘
                                  │                  │
                                  ▼                  ▼
                         ┌────────────────┐  Current regime label
                         │StudentTCopula  │  (calm / volatile / crisis)
                         │(copula.py)     │
                         └────────────────┘
```

### Sub-Models

#### `SectorBetaAnalyzer` (`src/risk/sector_beta.py`)
- Computes `beta[i][j] = cov(returns_i, returns_j) / var(returns_j)` for all sector pairs
- Dual-window estimation: 1-year and 3-year; `beta_window` config selects `"short"`, `"long"`, or `"average"`
- Identifies which assets belong to each sector via `sector_map`

#### `DCCGARCHModel` (`src/risk/dcc_garch.py`)
- **Stage 1**: Fit univariate GARCH(1,1) per asset → standardized residuals
- **Stage 2**: DCC recursion: `Q_t = (1−a−b)Q̄ + a·εε' + b·Q_{t-1}`; `R_t = diag(Q_t)^{-½} Q_t diag(Q_t)^{-½}`
- **MLE**: L-BFGS-B optimizer on DCC parameters `(a, b)` with constraint `a + b < 1`
- Produces **calm-regime correlation** (long-run `Q̄`) and **stress-regime correlation** (elevated DCC)

#### `StudentTCopula` (`src/risk/copula.py`)
- Kendall's tau → Pearson correlation matrix
- Fits degrees-of-freedom `ν` via MLE on empirical ranks
- `conditional_simulate(shocked_sectors, n_samples)`: draws from MVT, conditions on shocked-sector quantiles, back-transforms via empirical quantile function
- Used when `scenario.use_copula = True` to capture tail co-movement beyond linear correlation

#### `MarketRegimeDetector` (`src/risk/regime_detection.py`)
- Gaussian HMM (via `hmmlearn`) on rolling 21-day realized volatility
- Post-fit: states sorted by ascending mean conditional volatility → labels: `calm`, `volatile`, `crisis`
- Current regime read from last observation; badge displayed in UI

#### `LSEGSectorFetcher` (`src/data/lseg_sectors.py`)
- Primary: LSEG Data Library TRBC classification at `economic` / `business` / `industry` level
- Fallback: yfinance `info["sector"]` for stocks without LSEG access
- Accessed via `DataManager.get_sector_classifications()`

### Correlation Selection Logic (`_select_correlation`)

```python
if regime == "crisis" and dcc available:
    use DCC stress correlation
elif regime == "calm" and dcc available:
    use DCC calm correlation
elif copula available:
    use copula-implied correlation
else:
    use identity (no correlation propagation)
```

### Stress Propagation Algorithm

```
1. Identify directly shocked sectors from scenario.shocked_sectors
2. Select correlation matrix via _select_correlation()
3. For each non-shocked asset i:
       propagated_shock[i] = Σ_j (beta[i][j] × sector_shock[j] × correlation[i][j])
4. If use_copula: overlay copula conditional simulation for tail scenarios
5. Compute portfolio P&L: Σ_i (weight[i] × value × total_shock[i])
6. Assign role to each asset: "shocked" | "propagated" | "hedged" | "neutral"
```

### Pre-defined Scenarios (`DEFAULT_SCENARIOS`)

| Scenario | Shocked Sectors | Copula | DCC | Regime |
|----------|----------------|--------|-----|--------|
| Tech Selloff | Technology −25% | ✓ | ✓ | ✓ |
| Commodity Crash | Energy −30%, Materials −20% | ✓ | ✓ | ✓ |
| Rate Spike | Utilities −20%, Real Estate −25%, Finance +5% | ✗ | ✓ | ✓ |
| EM Risk-Off | Emerging Markets −30% | ✓ | ✓ | ✓ |
| Commodity Boom | Energy +25%, Materials +20% | ✗ | ✓ | ✓ |
| Financial Crisis | Finance −40%, Real Estate −35% | ✓ | ✓ | ✓ |
| Full Market Crash | All sectors −30% to −40% | ✓ | ✓ | ✓ |

### UI Flow

```
[Config Expander]  →  [Fit Engine Button]
        │
        ▼
Regime Badge  |  Beta Matrix Heatmap  |  DCC Correlation Heatmap
        │
        ▼
[Run Scenario Dropdown]  →  [Run All Scenarios Button]
        │
        ▼
Waterfall chart (per-asset P&L contribution)
Top Losers table  |  Hedge Candidates table
```

### Graceful Degradation

Each sub-model failure is isolated:

| Failure | `dcc_fail_action` / `*_fail_action` | Behavior |
|---------|--------------------------------------|----------|
| DCC GARCH fit fails | `"warn"` | Log warning, skip DCC; fall through to copula |
| Copula fit fails | `"warn"` | Log warning, skip copula; use identity matrix |
| Regime detection fails | `"warn"` | Log warning; treat as `"calm"` |

Set any `*_fail_action = "raise"` in `SectorStressConfig` to surface errors instead.

---

## File Map

```
app/
└── pages/
    └── 4_Stress_Testing.py          ← UI entry point

src/
├── simulation/
│   ├── scenarios.py                 ← StressTester, HISTORICAL_SCENARIOS (legacy)
│   ├── historical_scenarios.py      ← HistoricalStressor, HISTORICAL_SCENARIOS (actual)
│   ├── monte_carlo.py               ← MonteCarloSimulator
│   └── sector_stress.py             ← SectorStressEngine, DEFAULT_SCENARIOS
├── risk/
│   ├── sector_beta.py               ← SectorBetaAnalyzer
│   ├── dcc_garch.py                 ← DCCGARCHModel
│   ├── copula.py                    ← StudentTCopula
│   └── regime_detection.py          ← MarketRegimeDetector
└── data/
    ├── data_manager.py              ← get_sector_classifications()
    └── lseg_sectors.py              ← LSEGSectorFetcher (TRBC + yfinance fallback)
```

---

## Configuration Dataclasses

All engine behaviour is controlled via dataclasses with explicit `field(default=...)` declarations (never class-attribute or argument defaults):

```python
SectorStressConfig(
    beta_config=SectorBetaConfig(),
    dcc_config=DCCGARCHConfig(),
    copula_config=CopulaConfig(),
    regime_config=RegimeConfig(),
    portfolio_value=1_000_000.0,
    min_weight_threshold=0.001,
    dcc_fail_action="warn",
    copula_fail_action="warn",
    regime_fail_action="warn",
    beta_window="average",         # "short" | "long" | "average"
)

HistoricalStressorConfig(
    min_data_points=5,
    beta_estimation_days=252,
    beta_fallback_value=1.0,
    beta_clip_min=-3.0,
    beta_clip_max=3.0,
    min_pre_crisis_days=30,
)
```

---

## Data Flow Summary

```
Session State
    │
    ├── returns (DataFrame)    ──────────────────────────────────┐
    ├── weights (ndarray)      ──────────────────────────────┐   │
    ├── portfolio_value (float)─────────────────────────┐    │   │
    └── tickers (list)         ─────────────────────┐   │    │   │
                                                    │   │    │   │
Tab 1 ── HistoricalStressor ─────────────────── tickers weights value  │
Tab 2 ── MonteCarloSimulator ────────────────────────── weights  │  returns
Tab 3 ── StressTester ───────────────────────────────── weights value  │
Tab 4 ── SectorStressEngine.fit() ───────────────────────────────── returns
              └─ .run_stress(scenario, holdings) ─── weights value
```
