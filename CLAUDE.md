# CLAUDE.md

> Last verified against the codebase: 2026-07-07, commit `2c57ffd32aa1723ae29e2c4ecf924323d30f7370`.
> This file is not self-maintaining. If you change a module described below, re-verify the relevant line before trusting it — a stale entry here is worse than no entry.

## What this is

PortfolioOptimizer is a Streamlit web app + Python library for quantitative portfolio construction and risk analysis: multi-source price/fundamentals ingestion, 7 portfolio optimization methods (including HRP and Black-Litterman), VaR/GARCH/DCC-GARCH/copula risk analytics, historical and macro stress testing, Monte Carlo simulation, factor analysis, PDF reporting, a standalone stock-valuation pipeline, and a newer "Portfolio Builder" sector-neutral ranking feature. It has meaningful Indonesian-market (IDX, `.JK` tickers) support alongside US equities. Built for a solo/small-team quant workflow, not a regulated production trading system — treat it as a calculation-assistance tool (see README.md's disclaimer).

## Directory map

| Path | Purpose |
|---|---|
| `app/Home.py`, `app/pages/*.py` | Streamlit multipage UI. Page 1 (Portfolio Input) also owns Presets (merged in) as a tab. Page 10 is the Portfolio Builder feature below. |
| `src/portfolio_builder/` | Sector-neutral ranking + correlation network (MST) + HHI/Sharpe metrics feature. See [architecture.md#portfolio-builder](docs/architecture.md#portfolio-builder). |
| `src/data/` | Multi-source price fetch (yfinance→AlphaVantage→TwelveData→FMP), pickle cache, sector classification (TRBC/GICS), FRED/Trading-Economics macro data. See [architecture.md#data](docs/architecture.md#data). |
| `src/optimization/` | `PortfolioOptimizer` (7 methods), `HRPOptimizer`, `BlackLittermanModel`, `PortfolioConstraints` (position limits, sector caps, turnover trading-band). See [architecture.md#optimization](docs/architecture.md#optimization). |
| `src/risk/` | `RiskMetrics`, `VaRCalculator`, `GARCHModel`, `DCCGARCHModel` (sector-level), `StudentTCopula`, `MarketRegimeDetector` (HMM), `LeontifContagionEngine`, sector/stock beta. See [architecture.md#risk](docs/architecture.md#risk). |
| `src/simulation/` | `MonteCarloSimulator` (4 methods), legacy uniform-shock `StressTester`, real-per-stock `HistoricalStressor`, `SectorStressEngine`, `MacroStressEngine` (Leontief). See [architecture.md#simulation](docs/architecture.md#simulation). |
| `src/valuation/stock_valuer.py` | 3-stage fundamentals pipeline: Magic Formula screen → multi-factor score (0-100) → reverse DCF. See [architecture.md#valuation](docs/architecture.md#valuation). |
| `src/factors/` | Fama-French 3/5-factor, style factors, Brinson attribution, factor risk decomposition. Not deep-audited — see [architecture.md#other](docs/architecture.md#other). |
| `src/portfolio/` | `HoldingsTracker`, `PositionCalculator`, `PortfolioRebalancer`, `DCAScheduler`, `PerformanceAttributor`. Not deep-audited — see [architecture.md#other](docs/architecture.md#other). |
| `src/reports/` | PDF generation (`ReportGenerator`, `ReportChartGenerator`). Not deep-audited — see [architecture.md#other](docs/architecture.md#other). |
| `src/utils/` | `preset_manager.py` (portfolio preset persistence, no network dependency), `settings_manager.py`, logging, helpers. |
| `docs/architecture.md` | Full module-by-module detail — read on demand, not by default. |

## Load-bearing decisions

- **Portfolio Builder's ranking is one composite z-score, not sequential filter stages** — a hard filter would double-count correlated factors and can collapse the eligible universe; z-scoring is sector-neutral (within-sector mean/std) so structurally different sectors never dominate the rank. (`src/portfolio_builder/ranking.py`)
- **FF3+CMA regression (`ff5_overlay.py`) is a strictly post-hoc display overlay** — takes a constructed portfolio's returns as input, never feeds `ranking.py`'s composite score or any backtest objective. Grep-verifiable: zero references to `ff5_overlay` inside `ranking.py`.
- **Portfolio Builder's correlation network uses ticker-level plain `.corr()`, not `DCCGARCHModel` extended to ticker level** — `DCCGARCHModel` is fit at sector count (~11) in production; the portfolio_builder code's own stated reason for not extending it to ticker level is "an unresolved convergence-misreport risk," but independent audit of `src/risk/dcc_garch.py` found a hard `ConvergenceError` guard instead, not a documented misreport issue — the underlying justification doesn't currently trace to anything verifiable in `dcc_garch.py` itself. The decision (plain `.corr()` at ticker level) still stands; the cited reason is unverified. See [architecture.md#risk](docs/architecture.md#risk) for the full note.
- **Portfolio Builder's Sharpe estimate (`expected_return_estimate()`-style modeled returns, and the Sharpe calculation itself) is display-only, isolated from `ranking.py`** — zero references from `ranking.py`, grep-verifiable. Separately, `src/risk/metrics.py` has its **own, unrelated** `sharpe_ratio()` with no `lookback_days`/`SharpeConfig` concept at all — two independent Sharpe implementations coexist in this codebase; know which one a page uses.
- **Sharpe's return and volatility legs share one trailing 3-year (756-day) window, both annualized** — a prior 1-year-only design and then a current-day-only DCC-GARCH volatility snapshot both caused period mismatches that silently distorted Sharpe (confirmed via synthetic-data checks); fixed by trailing-averaging the model's own conditional-variance path over the same window as the return leg. (`src/portfolio_builder/metrics.py`)
- **Three unrelated mechanisms are all called "turnover"** — don't conflate them: (1) `src/optimization/constraints.py`'s `turnover_enabled`/`reduction_pct`/`increase_pct` weight trading-band (real, tested, actively used by all optimizer backends); (2) that same file's `max_turnover`/`get_turnover_constraint()` L1 cap (defined, appears unused/dead); (3) `src/portfolio_builder/ranking.py`'s `TurnoverConfig.entry_percentile`/`stay_percentile` (an unset placeholder for stock selection — see Known Placeholders below).
- **Two independent HRP implementations exist** — `src/optimization/hrp.py`'s `HRPOptimizer` (Ledoit-Wolf shrinkage by default) and `src/optimization/optimizers.py`'s `_optimize_hrp` (simpler, no shrinkage, hardcoded single-linkage). Neither calls the other.
- **Preset loading populates `current_holdings` via a live price fetch, not just `weights`/`portfolio_value`** — `src/utils/preset_manager.py::apply_preset_to_state()` only ever set `tickers`/`weights`/`portfolio_value`/`settings`/`loaded_preset_id`; it deliberately still does not touch `current_holdings`. `app/pages/1_Portfolio_Input.py`'s `_load_preset_and_populate_holdings()` wraps it and additionally derives `shares = weight * portfolio_value / current_price` — kept out of `preset_manager.py` on purpose to keep that module free of a network/`DataManager` dependency.
- **Portfolio Presets is a tab on Portfolio Input, not a separate page** — `app/pages/9_Portfolio_Presets.py` no longer exists; merged in because preset-loading and holdings-editing are the same workflow.

## Known placeholders — do not mistake these for real, load-bearing values

- **`src/portfolio_builder/ranking.py::TurnoverConfig`**: `entry_percentile=0.0`, `stay_percentile=0.0` — unset, TBD by backtest. `__post_init__` logs a `logger.warning` when both are still 0.0/0.0, explicitly stating this is "NOT a safe conservative value, it is unset." Nothing in the codebase currently consumes these for a real entry/exit decision.
- **`src/portfolio_builder/ranking.py::PointInTimeLagConfig.idx_lag_months = 7`**: inline `# TODO: validate against real observed IDX filing timestamps once available` — a guessed constant, not yet checked against real data. (`us_lag_months = 6` is not similarly flagged.)
- **`src/data/lseg_sectors.py::IDX_SECTOR_OVERRIDES["TLKM.JK"]`**: applied unconditionally in code, but carries an inline `# Confirm vs LSEG assignment` comment — i.e. shipped before the override was actually confirmed correct.
- **`compute_dcf_gap()` sign convention** (`src/portfolio_builder/fetch.py`): integration-verified against one real ticker (KO) as of this writing — confirmed correct in that case, but not validated across a broad sample.
- **NOT a real placeholder, despite resembling the pattern above**: `src/optimization/constraints.py`'s turnover trading-band (`reduction_pct=0.50`/`increase_pct=0.30`) is a real default with regression tests (`tests/test_turnover_constraint.py`), not a TBD value. Don't lump it in with `ranking.py`'s `TurnoverConfig` above.
- **There is no `min_overlap_days` / crisis-mode-correlation-fallback mechanism anywhere in `src/risk/`** — confirmed by exhaustive grep. If you're looking for "a shorter window during crisis periods," it doesn't exist; the closest real things are `src/portfolio_builder/fetch.py::FetchConfig.min_correlation_overlap_days = 60` (a general reliability floor, unrelated to crisis regimes) and `src/risk/regime_detection.py::MarketRegimeDetector.get_regime_correlation()` (averages DCC matrices over dates in a regime, identity-matrix fallback below 5 aligned observations — a real, non-placeholder guard).
- **`src/risk/sector_beta.py::compute_stock_betas_vs_portfolio_sectors()`'s own docstring** claims to be "the primary stock-beta path used by `SectorStressEngine.fit()`" — this is stale/inaccurate; production actually uses `compute_all_stock_betas()` from `src/risk/stock_sector_beta.py`. Don't repeat the docstring's claim.

## Absolute constraints

- Streamlit + the existing Python backend only. No React, Redis, graph database, WebSockets, or 3D rendering anywhere in this codebase (confirmed: no `package.json`, no such dependencies in `requirements.txt`, no matching imports).
- Do not modify the existing ranking, DCC-GARCH, or sector-normalization modules' core algorithms as a side effect of unrelated work — extend via new config fields/functions the way `src/portfolio_builder/` does, not by rewriting what's already reviewed and merged.

## Where to go next

For full module-by-module detail — dataclass listings, exact function signatures, and the reasoning behind each non-obvious choice (why HRP over ticker-level DCC-GARCH, why MST-plus-threshold over a bare threshold, why sector-neutral z-scoring specifically) — see [`docs/architecture.md`](docs/architecture.md). Read the section for the module you're about to touch; this file is an index, not a substitute for reading the code you're changing.
