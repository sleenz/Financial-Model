# PortfolioOptimizer

An advanced quantitative portfolio optimization and risk management system with a Streamlit web interface. Provides institutional-grade tools for portfolio construction, risk analytics, stress testing, factor analysis, stock valuation, and PDF reporting — with meaningful support for both US equities and the Indonesian market (IDX, `.JK` tickers).

> **Disclaimer:** This tool is a calculation-assistance aid. It is not intended as the sole basis for financial decisions. Always do your own research (DYOR).

For internals — module-by-module design detail, formulas, and the reasoning behind non-obvious choices — see [`docs/architecture.md`](docs/architecture.md). That document (and its companion [`CLAUDE.md`](CLAUDE.md)) is aimed at whoever is next modifying the code; this README is aimed at getting the app running.

---

## Quick Start

### 1. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```env
ALPHA_VANTAGE_KEY=your_key_here
TWELVE_DATA_KEY=your_key_here
FMP_KEY=your_key_here
```

yfinance (the primary data source) works without any key. The other sources are fallbacks used only if yfinance fails. Macro data additionally benefits from a Trading Economics key (`TE_API_KEY`) and a FRED key (`FRED_API_KEY`) — both optional, with reduced coverage if omitted.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the web interface

```bash
streamlit run app/Home.py
```

---

## Features

| Capability | Description |
|---|---|
| Data Ingestion | Multi-source fallback: yfinance → Alpha Vantage → Twelve Data → FMP, with disk caching |
| Portfolio Optimization | 7 algorithms: Max Sharpe, Min Vol, Risk Parity, HRP, Max Diversification, Max Return, Equal Weight |
| Black-Litterman | Market-equilibrium prior + investor views (absolute/relative), Bayesian posterior |
| Risk Analytics | VaR/CVaR (historical, parametric, Cornish-Fisher), GARCH, drawdown family, Sharpe/Sortino/Calmar/Omega |
| Deep Risk Analysis | Tail risk (Jarque-Bera, QQ plot), Monte Carlo VaR, component/marginal VaR, Effective Number of Bets via PCA |
| Hedging Effectiveness | Beta classification, risk contribution decomposition, diversification benefit waterfall |
| Historical Stress Testing | 7 crisis scenarios replayed with actual per-stock returns; beta-scaled proxy for stocks that predate the event |
| Sector Shock Stress Test | Per-stock sector-relative beta (ETF OLS with circularity correction); DCC-GARCH dynamic correlations; Student-t copula tail dependence; HMM regime-conditioned correlation selection |
| Macro Contagion Stress Test | Leontief input-output contagion model; macro sensitivity matrix (Trading Economics/FRED/yfinance); spectral-radius cascade risk |
| Monte Carlo Simulation | 4 methods: GBM, block bootstrap, Student-t, jump-diffusion |
| Factor Analysis | Fama-French 3/5-factor, style factors (momentum/value/quality/low-vol/size), Brinson attribution |
| Portfolio Tracking | Holdings entry (tickers + shares) or manual ticker list; diversity metrics (HHI, Gini, effective stocks) |
| Portfolio Presets | Save/load/rename named portfolio snapshots; loading a preset populates editable holdings at current market prices |
| Portfolio Builder | Sector-neutral 4-factor ranking, correlation network (MST) with semantic zoom, HHI/diversification + period-matched Sharpe |
| Rebalancing | Drift detection, trade recommendations with share counts, DCA scheduler |
| Reporting | PDF generation with 6 chart types and 3 report templates |
| Stock Valuation | Standalone 3-stage pipeline: Magic Formula screen → Multi-Factor Score → Reverse DCF |

---

## Repository Layout

```
PortfolioOptimizer/
|
+-- app/                                  # Streamlit multi-page application
|   +-- Home.py                           # Landing page, session state init
|   +-- pages/
|       +-- 1_Portfolio_Input.py          # Holdings entry, manual tickers, and Presets (3 tabs)
|       +-- 2_Optimization.py             # Optimization + rebalancing UI
|       +-- 3_Risk_Analytics.py           # Risk dashboard + deep analysis
|       +-- 4_Stress_Testing.py           # Historical, Sector Shock, Macro Contagion
|       +-- 5_Monitoring.py               # Rebalancing + attribution + DCA
|       +-- 6_Factor_Analysis.py          # FF factors + style + decomposition
|       +-- 7_Reports.py                  # PDF report generation UI
|       +-- 8_Stock_Valuation.py          # Magic Formula / multi-factor / reverse DCF UI
|       +-- 10_Portfolio_Builder.py       # Sector-neutral ranking + correlation network
|
+-- src/                                  # Core library
|   +-- portfolio_builder/                # Ranking, correlation network, metrics (see docs/architecture.md)
|   +-- data/                             # Multi-source fetch, cache, sector/macro data
|   +-- optimization/                     # PortfolioOptimizer, HRP, Black-Litterman, constraints
|   +-- risk/                             # RiskMetrics, VaR, GARCH/DCC-GARCH, copula, regimes, contagion
|   +-- simulation/                       # Monte Carlo, historical/sector/macro stress engines
|   +-- valuation/                        # stock_valuer.py — 3-stage valuation pipeline
|   +-- factors/                          # Fama-French, style factors, attribution
|   +-- portfolio/                        # Holdings tracking, rebalancing, DCA, attribution
|   +-- reports/                          # PDF generation
|   +-- utils/                            # Presets, settings, logging, helpers
|
+-- tests/                                # pytest suite (optimization constraints, presets, sector beta, ...)
+-- docs/architecture.md                  # Full module-by-module design detail
+-- CLAUDE.md                             # AI-agent-facing index (load-bearing decisions, known placeholders)
+-- requirements.txt                      # Python dependencies
+-- .env.example                          # API key template
+-- README.md                             # This file
```

Most `src/` modules also ship a runnable smoke test behind `if __name__ == "__main__":` (e.g. `python -m src.portfolio_builder.ranking`) in addition to, or instead of, `pytest` coverage under `tests/` — see `docs/architecture.md` for which modules use which.

---

## Known Issues

| Severity | Location | Issue |
|---|---|---|
| Low | `app/pages/3_Risk_Analytics.py` (EWMA Volatility panel) | Plots `(returns*weights).sum(axis=1).rolling(20).std()` instead of the computed `ewma_vol` series it fetches just above. Cosmetic mislabel only — the correct series is fetched, just not the one plotted. |

---

## License / Contributing

No license file is present in this repository as of this writing — confirm usage terms with the repository owner before redistributing.
