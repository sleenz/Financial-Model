# Architecture

> Last verified against the codebase: 2026-07-07, commit `2c57ffd32aa1723ae29e2c4ecf924323d30f7370`.
> This document is not self-maintaining. If a change touches a module described below, re-verify that section against the actual file before trusting it — do not assume it's still accurate just because it's written down.

This is the detailed companion to [`CLAUDE.md`](../CLAUDE.md)'s directory map. Each section below matches one row of that map. Read the section for the module you're about to touch; don't read this file end-to-end by default.

---

<a id="portfolio-builder"></a>
## `src/portfolio_builder/` — the "Portfolio Builder" feature

A self-contained sector-neutral stock ranking + correlation-network + metrics feature, built as Phases 0-6 of one build spec. It reuses the pre-existing `src/data/`, `src/valuation/`, `src/optimization/hrp.py`-style correlation, and `src/factors/fama_french.py`, but owns its own ranking algorithm, cache, and rendering config. Wired into the UI at `app/pages/10_Portfolio_Builder.py`.

### `ranking.py` (474 lines) — sector-neutral composite ranking

`RankingEngine` computes a 4-factor composite score, sector-neutral (z-scored within each sector, never across the full universe).

- `FactorConfig.factors: tuple = ("earnings_yield", "roc", "momentum", "dcf_gap")` (`ranking.py:28`) — closed list, comment explicitly says don't add a 5th factor here.
- `CompositeWeights` — equal-weight (0.25 each) starting point (`ranking.py:34-37`); comment: "Recalibrated by backtest — never hand-tuned, never fed by `expected_return_estimate()` or Sharpe."
- `TurnoverConfig(entry_percentile: float = 0.0, stay_percentile: float = 0.0)` (`ranking.py:43-69`) — **placeholder, not a real value** (see CLAUDE.md's Known Placeholders). `__post_init__` raises if `stay_percentile > entry_percentile`, and logs a `logger.warning` when both are still at the 0.0/0.0 default, explicitly stating "this is NOT a safe conservative value, it is unset." This is a completely different concept from `src/optimization/constraints.py`'s `turnover_enabled`/`reduction_pct`/`increase_pct` weight trading-band — that one is a real, tested, actively-used mechanism; this one is an unset stock-selection knob. Do not conflate them.
- `PointInTimeLagConfig(us_lag_months: int = 6, idx_lag_months: int = 7)` (`ranking.py:73-76`) — IDX's 7-month lag has an inline `# TODO: validate against real observed IDX filing timestamps once available`.
- `RankedStock` dataclass has a `factor_coverage: float` field (`ranking.py:80-108`) — fraction of the 4 factors that were real (non-NaN) *before* neutral-fill, so a composite built from 1 real z-score + 3 filled zeros doesn't look identical in the UI to one built from 4 real z-scores. The old `rank_tier` ("high"/"mid"/"low" bucket) field is gone — replaced by `heat_color.py`'s continuous gradient. No code in this module actually constructs `RankedStock` instances yet — `compute_factor_zscore`/`compute_composite_score`/`apply_point_in_time_lag` are the three implemented methods; assembling a full ranked list end-to-end is left to the next consumer.
- `RankingEngine.compute_factor_zscore(raw_factor_values, sector_map) -> pd.Series` (`ranking.py:126-193`): z = (x - sector_mean) / sector_std, `ddof=0` (population std — the sector's own members ARE the reference population, not a sample). A sector with zero variance gets z=0.0 for its members; a ticker's own NaN input always stays NaN even inside a degenerate sector (regression-tested — an earlier version conflated the two cases).
- `RankingEngine.compute_composite_score(zscores, weights) -> tuple[composite, factor_coverage]` (`ranking.py:195-247`): weighted sum, no hard filters — a missing factor is treated as neutral (0.0), never drops the ticker.
- `RankingEngine.apply_point_in_time_lag(fundamentals, as_of_date, market) -> (kept, excluded)` (`ranking.py:249-290`): excludes fundamentals whose `fiscal_year_end` falls within the lag window; **deviates from the original spec's `-> pd.DataFrame` type hint** — returns a `(kept, excluded)` tuple instead, since the excluded rows must be visible, not silently dropped.
- **Isolation guarantee** (module docstring, `ranking.py:10-12`): `expected_return_estimate()` and Sharpe are display-only and have zero references in this file — grep-verifiable (`grep -c expected_return_estimate src/portfolio_builder/ranking.py` → 0 matches outside the module docstring's own mention of the name).

### `fetch.py` (514 lines) — on-demand fetch + interim scoring

`PortfolioDataLayer` binds together the pre-existing data sources: `DataManager` (prices/sectors) + `src.valuation.stock_valuer.multi_factor_score`/`reverse_dcf` (fundamentals DataManager doesn't fetch).

- `FetchConfig` (`fetch.py:48-59`): `price_history_days=400` (>252+21 trading days needed by stock_valuer's momentum/SMA-200 calcs), `min_correlation_overlap_days=60` (below this, a correlation is unreliable and is left empty rather than computed on too few overlapping return observations — this is the fetch-time overlap guard the network layer's `_row_is_complete` check later relies on), `dcf_wacc=0.10`, `min_data_quality_score=0.0` (guards against `multi_factor_score()` "succeeding" with an all-zero dict when every sub-metric failed to fetch).
- `compute_dcf_gap(dcf: dict) -> Optional[float]` (`fetch.py:91-126`): `dcf_gap = -growth_premium`. **Sign integration-verified against real data**, not just unit-tested: Coca-Cola (KO), priced for +31.8% implied growth against a -17.8% trailing 3yr FCF CAGR (`growth_premium=+0.496`, an "Extreme" `reverse_dcf` verdict), produced `dcf_gap=-0.496` and the *lowest* z-score in a real 4-ticker peer comparison — confirming the sign matches the design intent (a stock priced for *less* growth than its own trend should score *higher*). Returns `None` (not `0.0`) when `growth_premium` is itself `None` — a missing input propagates as missing.
- Correlation: `compute_universe_entries()` (`fetch.py:248-291`) computes one real NxN correlation matrix via plain `returns.corr()` (not refit per ticker), gated by `min_correlation_overlap_days`; below that threshold, `correlation_row` is left empty for the whole refresh and logged, not silently zeroed.
- `_build_entry()` (`fetch.py:182-245`) raises rather than caching a placeholder score on total fundamentals failure or a `data_quality_score` at/below `min_data_quality_score` — both are treated as "no data," not "genuinely scores 0."
- `OnDemandFetcher.get_or_fetch(ticker)` (`fetch.py:294-340`): cache-hit returns immediately; cache-miss fetches, scores, and (if `CacheConfig.on_demand_join_cache`) joins the cache with an empty `correlation_row`, deferred to the next `run_nightly_refresh()` — see `network.py`'s cold-cache handling below for why this matters.

### `cache.py` (311 lines) — SQLite persistence

`UniverseCache`, a pure persistence layer — computation lives in `fetch.py` and is injected (not imported at module load, to avoid a circular import). Different path/class/schema from the pre-existing `src/data/cache.py::DataCache` — no collision, but easy to confuse the two by name.

- `CacheConfig(cache_path="data/portfolio_builder_cache.db", cache_ttl_hours=24, on_demand_join_cache=True)` (`cache.py:35-41`).
- `RankedUniverseEntry` dataclass (`cache.py:44-54`): `ticker, sector, market, composite_score, factor_zscores, correlation_row, computed_at`.
- `UniverseCache` wraps one shared `sqlite3.connect(..., check_same_thread=False)` connection behind a single `threading.RLock()` (`cache.py:76`) — comment explains this is load-bearing: Streamlit shares one `UniverseCache` across request threads, and `check_same_thread=False` alone does *not* make concurrent access safe (observed failure modes noted in the comment: "no more rows available", "cannot commit", even a raw `SystemError`). `RLock` (not `Lock`) specifically because `run_nightly_refresh()` re-enters via its own call to `self.upsert()` while already holding the lock.
- `get(ticker)` returns `None` on a TTL-expired or malformed/corrupted row (treated as a cache miss, not an exception) — triggers the on-demand fetch path.
- `run_nightly_refresh(universe, data_layer=None, compute_fn=None)` — both injectable; default lazily imports `fetch.py`'s `build_default_data_layer`/`compute_universe_entries`.

### `network.py` (733 lines) — correlation network (MST) + semantic zoom

Reconstructs the ticker×ticker correlation matrix from `UniverseCache`'s per-ticker `correlation_row` values (or from a directly-supplied live correlation DataFrame), builds a Minimum Spanning Tree via the Mantegna (1999) distance transform, and exposes both a full ticker-level MST and a sector-supernode MST for "semantic zoom."

- **Why plain `.corr()`, not DCC-GARCH, at ticker level** (module docstring, `network.py:4-10`, and `fetch.py:11-17`): the existing `DCCGARCHModel` (`src/risk/dcc_garch.py`) is fit at *sector* count and has an unresolved convergence-misreport issue (see the `src/risk/` section below); running it at ticker count would multiply that risk for a feature where a wrong number is invisible in the UI. This was a deliberate scope decision, revisit-only-if the MST clusters don't line up with real sector groupings — not yet revisited.
- `compute_distance_matrix()` (`network.py:161-169`): Mantegna transform `d = sqrt(2*(1-rho))` — a fixed mathematical definition, not a config field (same treatment as the Sharpe formula itself). `correlation_from_distance()` (`network.py:242-250`) is its exact algebraic inverse (`rho = 1 - d^2/2`), so downstream code that needs correlation back (e.g. edge coloring) doesn't have to carry a second parallel matrix.
- `build_correlation_matrix(cache, tickers, config)` (`network.py:106-158`): excludes (and reports, via `logger.warning` + the returned `excluded_tickers` list) any ticker with no cache entry or an incomplete `correlation_row` — never silently drops a ticker or lets a `None`-containing row reach `nx.minimum_spanning_tree` (which would crash on a NaN edge weight). `_row_is_complete()` (`network.py:92-103`) is the completeness check: right length AND every entry non-`None`.
- **Cold-cache handling** (`build_semantic_zoom_network`'s `correlation` parameter, `network.py:396-458`): with no nightly refresh job actually scheduled in a given deployment, every first-time ticker's `correlation_row` is empty (per `OnDemandFetcher`'s deferred-fill contract), so `build_correlation_matrix` alone would exclude every new ticker and raise "no tickers had usable correlation data" for every new user. The page (`app/pages/10_Portfolio_Builder.py`) instead passes an already-computed live correlation matrix from price data it fetched for another purpose, bypassing the cache dependency entirely. This was a real production bug caught and fixed, not a hypothetical.
- `NetworkConfig(mst_algorithm="kruskal", sector_aggregation="average", require_full_correlation_row=True)` (`network.py:54-60`).
- `NetworkStyleConfig` (`network.py:253-284`): edge strength is a diverging **color gradient** (not opacity — opacity is now one fixed value, `edge_opacity=1.0`, applied to every edge). `edge_color_for_correlation()` (`network.py:305-317`) linearly interpolates in RGB space between `edge_color_negative` (`#4682b4`, "steelblue") at rho=-1, `edge_color_neutral` (`#d3d3d3`, "lightgray") at rho=0, and `edge_color_positive` (`#ff7f50`, "coral") at rho=+1. Node tier colors (`node_color_top`/`mid`/`bottom` = green/orange/red) reuse the same tri-tier percentile thresholds (0.67/0.33) as the Ranked List's own emoji.
- `CorrelationNetworkConfig(always_include_mst=True, positive_threshold=0.30, hedge_threshold=-0.30)` (`network.py:334-356`): governs which *non-MST* edges get drawn alongside the always-present MST. Two independent same-signed thresholds rather than one `|correlation|` cutoff, because "strongly correlated" and "strongly anti-correlated / hedge-like" are different things a user looks for, not two ends of one slider. `filter_edges_by_threshold()` (`network.py:358-393`) is a pure re-filter over an already-computed distance matrix + MST — no recomputation — which is what lets a UI threshold slider be a cheap client-side operation on every Streamlit rerun instead of a re-fetch/re-fit.

### `metrics.py` (746 lines) — sector exposure (HHI) + period-matched Sharpe

Two independent, display-only metric groups.

- `compute_sector_exposure(weights, sector_map, config) -> SectorExposureResult` (`metrics.py:69-106`): HHI = sum of squared sector weights. A zero-weight ticker's sector is dropped before computing "sectors held" — an unfunded watchlist ticker must not inflate the sector count (regression-tested: this was a real bug where the same real holding scored differently depending on which zero-share ticker happened to also be in the caller's list).
- `compute_diversification_rating(sector_exposure, ticker_mst) -> float` (`metrics.py:109-160`): fuses a sector-spread sub-score (rescaled HHI effective-N, Woerheide & Persson 1993) and a correlation-density sub-score (mean Mantegna MST edge distance / 2.0, Onnela et al. 2003) via their **geometric mean**, deliberately not arithmetic — either sub-score at 0 forces the whole rating to 0, so a portfolio strong on one axis but degenerate on the other can't get a misleadingly high blended score.
- `SharpeConfig` (`metrics.py:163-203`):
  - `lookback_days: int = 3 * 252` (756, 3 years) — **per explicit user instruction** to use a 3-year average rather than the original 1-year spec.
  - `risk_free_rate: Optional[float] = None` — deliberately has **no numeric default**; FRED-sourced risk-free rate is broken upstream (Phase 0 audit), so `compute_sharpe()` raises `ValueError` if this is still `None` rather than silently defaulting to 0.0 (which would overstate Sharpe). The page (`app/pages/10_Portfolio_Builder.py:60`) sets a manual constant (`_MANUAL_RISK_FREE_RATE = 0.045`) with a comment to update it periodically until FRED is fixed.
  - `volatility_source: str = "dcc_garch_trailing_average"` — closed choice, `compute_sharpe()` raises if changed to anything else. This was originally `"dcc_garch_current"` (single latest-day snapshot) until a user-reported "impossibly high Sharpe ratio" investigation confirmed via a controlled synthetic-data check that a current-day-only snapshot can diverge 2x+ from the trailing-window average whenever the current regime differs from that window.
- `compute_realized_return(portfolio_returns, lookback_days, annualization_days=252) -> float` (`metrics.py:206-248`): **annualizes (CAGR)** the trailing window rather than returning the raw cumulative total — `(1+cumulative)^(annualization_days/lookback_days) - 1`. At `lookback_days == annualization_days` (252, the historical 1-year case) this is identical to the old cumulative-return behavior; for the current 756-day (3-year) default, this keeps the figure on the same annual scale as the volatility leg instead of reporting a much larger multi-year total against a 1-year-scale volatility.
- `compute_dcc_garch_volatility_current()` (`metrics.py:251-313`) vs. `compute_dcc_garch_volatility_trailing()` (`metrics.py:316-403`): the former uses only the DCC-GARCH model's single latest fitted conditional covariance; the latter averages the model's own daily conditional *variances* (not volatilities — variance first, sqrt last, same principle as realized variance being built from squared returns) over the trailing `lookback_days` window. **Only the trailing version feeds `compute_sharpe()`** — the current-only version is kept for a legitimately different question ("what's volatility right now") but is no longer part of the Sharpe calculation. Neither ever calls `forecast_correlation()` at any horizon — a projected horizon would extrapolate past the disclosed, already-fitted history.
- Both DCC-GARCH volatility functions require **sector-level** weights aligned to `dcc_result.sector_names` (the DCC-GARCH engine is fit at sector level, never ticker level — same reason as `network.py`'s design decision above) and raise `ValueError` on a ticker-level/misaligned weights `Series` rather than silently misaligning.
- `compute_sharpe(realized_return, volatility, config) -> float`: `(realized_return - risk_free_rate) / volatility`. Raises on `risk_free_rate is None`, on `volatility_source` != the one supported value, and on `volatility <= 0`.

### `heat_color.py` (147 lines) — RdYlGn gradient for the ranked list

Replaces the old `rank_tier` ("high"/"mid"/"low", never actually implemented with any cutoff logic anywhere in the codebase — confirmed by grep before deletion) with a continuous `matplotlib` `RdYlGn` diverging colormap.

- `composite_score_to_color(score, low_bound, high_bound, colormap="RdYlGn") -> str` (`heat_color.py:46-70`): clips to `[low_bound, high_bound]`, normalizes to `[0,1]`, maps through `matplotlib.colormaps[colormap]` (**not** `matplotlib.cm.get_cmap`, which was removed by matplotlib 3.11.0 — the version pinned/installed here; using the removed function raises `AttributeError`). Raises `ValueError` if `high_bound <= low_bound`.
- `HeatColorConfig(colormap="RdYlGn", low_percentile=0.05, high_percentile=0.95)` (`heat_color.py:30-44`): bounds are meant to be computed from the **full cached universe** (the nightly batch job's output), not the current user selection — otherwise a mediocre stock could read as "good" purely because everything else in a small selection happens to be worse.

### `ff5_overlay.py` (232 lines) — FF3+CMA rolling regression (post-hoc only)

`FF3CMAOverlay` — rolling 3-factor (Mkt-RF, SMB, CMA; **HML and RMW dropped per spec**) OLS regression of a constructed portfolio's own returns against factor data.

- **Isolation guarantee** (module docstring, `ff5_overlay.py:5-9`): strictly post-hoc/display — takes a portfolio's returns as an *input*, produces regression diagnostics as an *output*. Never called from `ranking.py`, never feeds the composite score or a backtest objective (grep-checked).
- Reuses `src.factors.fama_french.get_factor_data()` (5-factor pull, yfinance/pandas_datareader with synthetic-data fallback) rather than reimplementing the fetch — narrows to just the 3 factors it needs.
- `FF3CMAConfig(factors=("Mkt-RF","SMB","CMA"), rolling_window_days=60, risk_free_col="RF", min_window_observations=20)` (`ff5_overlay.py:37-41`).
- `compute_rolling_exposures()` (`ff5_overlay.py:64-163`): rolling OLS via `np.linalg.lstsq`, skipping (and logging) any window with fewer than `min_window_observations` valid rows rather than fitting on too little data.

### UI wiring — `app/pages/10_Portfolio_Builder.py`

Assembles all six modules above into one page: ticker chips, ranked list (heat-colored score + editable share count + computed % weight), correlation network (sector overview with drill-into-sector detail), and a metrics panel (HHI/diversification + Sharpe).

- `_MANUAL_RISK_FREE_RATE = 0.045` (line 60) — see `SharpeConfig.risk_free_rate` above.
- `_CORRELATION_LOOKBACK_DAYS = 252` (line 75) — **deliberately decoupled** from `SharpeConfig().lookback_days` (756). Both features share the same fetched `hist_prices` DataFrame, but each slices its own trailing window from it; extending the Sharpe window (per explicit instruction) must not silently also extend the correlation network's/DCC-GARCH's effective sample window, since nobody asked to change that.
- Every network/data-fetch/DCC-GARCH-fit call is cached in `st.session_state`, keyed by the current ticker set only — a share-count edit changes % weight and the Sharpe/volatility numbers via plain arithmetic on rerun, never re-keys the cache or triggers a new fetch/fit (a visible call counter at the bottom of the page makes this observable).

---

<a id="data"></a>
## `src/data/` — data ingestion, caching, sector/macro data

### `data_manager.py` (521 lines) — `DataManager`, the central orchestrator

- Fallback source order is fixed at construction (`data_manager.py:54-60`): **YFinanceSource → AlphaVantageSource → TwelveDataSource → FMPSource**. `get_price_data()` (`data_manager.py:64-71, 131-164`) checks `DataCache` first, then tries sources in order, taking the **first non-empty result** (not a merge across sources); raises `DataSourceError` only if every source fails/returns empty (`data_manager.py:167-170`).
- Cache writes from `get_price_data()` always use `data_type="historical"` (`data_manager.py:182-184`) — a repo-wide grep found **zero** call sites anywhere that pass `data_type="intraday"`, so `DataCache`'s separate intraday TTL exists in code but is never actually exercised.
- `get_current_prices()`, `get_ohlcv_data()`, `get_market_caps()`, `get_sector_info()` **bypass the fallback chain entirely** — each imports `yfinance` directly and has no caching of its own (`data_manager.py:274-377`). Only `get_price_data()` goes through the 4-source chain + cache.
- `get_sector_classifications(tickers, level="economic", lseg_config=None)` (`data_manager.py:379-424`) delegates to `LSEGSectorFetcher` (see below), lazily imported to avoid a hard dependency on the optional `lseg-data` package.
- `set_source_priority(source_names)` (`data_manager.py:458-478`) lets a caller reorder `self.sources` post-construction; unnamed sources are appended at the end, not dropped.

### `sources.py` (521 lines) — the four fallback source adapters

- `YFinanceSource` (`sources.py:83-181`): always available (no API key), batches tickers in groups of `batch_size=5`, retries via `tenacity` (3 attempts, exponential backoff), sleeps 0.5-1.5s between batches.
- `AlphaVantageSource` / `TwelveDataSource` / `FMPSource` (`sources.py:184-521`): each reads its API key from an env var and marks itself permanently unavailable (`mark_unavailable()`, no automatic re-enable) if the key is missing or still the literal placeholder string from `.env.example` (e.g. `"your_alpha_vantage_key_here"`), or after hitting a `RateLimitError`. Per-ticker rate-limit sleeps: AlphaVantage 12s, TwelveData 8s, FMP 0.5s — comments cite each provider's free-tier call limits. FMP has no dedicated `RateLimitError` handling (only a generic `except Exception`), unlike the other two.

### `cache.py` (315 lines) — `DataCache`

- **File-based pickle cache** (not SQLite, not in-memory) — one `.pkl` file per key under `cache_dir` (env var `CACHE_DIR`, default `.cache`). Two TTLs: `ttl_intraday=3600s` and `ttl_historical=86400s` (env-overridable), but per `data_manager.py`'s usage above, only the historical TTL path is ever actually invoked in this codebase.
- This is a **separate cache from `src/portfolio_builder/cache.py`'s `UniverseCache`** (different storage mechanism entirely — pickle-on-disk here vs. SQLite there) — the two are unrelated and shouldn't be confused despite similar names.

### `lseg_sectors.py` (514 lines) — `LSEGSectorFetcher`

- Taxonomy: **TRBC** (Thomson Reuters Business Classification) is the source of truth (`TR.TRBCEconomicSector`/`BusinessSector`/`Industry` fields via the optional `lseg-data` package). yfinance's `.info["sector"]` returns **GICS** labels, normalized to TRBC via a hardcoded `GICS_TO_TRBC` map (`lseg_sectors.py:33-46`).
- 5-step fetch strategy (`lseg_sectors.py:253-364`): DataCache check → LSEG batch fetch (if the `lseg-data` package is installed and configured) → yfinance-per-ticker fallback for anything LSEG didn't resolve → `"Unknown"` for anything still unresolved → apply `IDX_SECTOR_OVERRIDES`.
- `IDX_SECTOR_OVERRIDES` (`lseg_sectors.py:54-61`) — a hardcoded 6-ticker table (PGEO.JK, GOTO.JK, BREN.JK, PGAS.JK, TLKM.JK, EMTK.JK) that always wins regardless of source, applied *after* LSEG/yfinance classification. **`TLKM.JK`'s entry carries an inline `# Confirm vs LSEG assignment` comment** (`lseg_sectors.py:59`) — i.e. applied unconditionally in code despite the comment flagging it as not yet confirmed. Treat as a known unvalidated assumption, same spirit as ranking.py's IDX lag TODO.
- In most environments (the proprietary `lseg-data` package isn't in default `requirements.txt` — see the commented-out line), `_LSEG_AVAILABLE` is `False` and every classification effectively goes straight to the yfinance-fallback + IDX-override path.

### `macro_data.py` (850 lines) — `MacroDataFetcher`

- Contrary to a surface reading of the module docstring ("from yfinance and FRED"), **the primary source for 7 of 9 default macro variables is actually Trading Economics** (`te_market`/`te_indicator`); yfinance and FRED are configured as *fallback* sources for some variables only. Exact primary/fallback pairing per variable (`macro_data.py:115-206`): DXY, VIX, IDR_USD, COAL fall back to yfinance; US_10Y, CPO, NICKEL fall back to FRED (`DGS10`, `PPOILUSDM`, `PNICKUSDM` respectively — the only 3 FRED series actually used, and only as fallbacks); BI_RATE and CHINA_PMI have **no fallback at all** — if Trading Economics fails or `TE_API_KEY` is unset, these two simply end up in `missing_variables`.
- `fetch()` (`macro_data.py:293-391`) never raises — every per-variable exception is caught and the variable is added to `missing_variables`, by design ("degrades gracefully" per its own docstring).
- Three optional-dependency import guards (`sqlite3`, `joblib`, `networkx`, `macro_data.py:43-62`) are defined but **never referenced anywhere else in the file** — vestigial, likely copy-pasted boilerplate.

### `validators.py` (331 lines) — `DataValidator`

- `validate_price_data()` runs 6 ordered checks per column (`validators.py:131-206`): length ≥ `min_data_points` (30) → missing-value % (≤5% fills via ffill/bfill, else the whole column is dropped as "critical") → zero-value % (warning only) → stale/constant-value streak (warning only) → outlier detection on returns (warning only) → negative-price check (critical, drops the column). Only "critical" issues cause a column to be dropped; everything else is a warning with the cleaned series substituted back in.

### Point-in-time lag — confirmed to NOT exist in `src/data/`

A repo-wide grep for "point-in-time"/"lag"/"PIT" inside `src/data/` returns zero matches. The fundamentals point-in-time lag concept (`PointInTimeLagConfig`, `us_lag_months=6`/`idx_lag_months=7`) lives **exclusively** in `src/portfolio_builder/ranking.py`. `src/data/`'s only IDX-specific logic is the hardcoded `IDX_SECTOR_OVERRIDES` sector-correction table above and passing `.JK`-suffixed tickers through to yfinance/LSEG as-is — no date-based fundamentals lag of any kind.

<a id="optimization"></a>
## `src/optimization/` — portfolio construction

### `optimizers.py` (532 lines) — `PortfolioOptimizer`, the main engine

- 7 methods registered in `method_map` (`optimizers.py:82-90`): `max_sharpe`, `max_return`, `min_volatility`, `max_diversification`, `risk_parity`, `hrp`, `equal_weight` — all via `scipy.optimize.minimize(method='SLSQP')` except `equal_weight` (analytical, `1/n`) and `hrp` (its own hierarchical-clustering code path, not SLSQP).
- Contains its **own independent HRP re-implementation** (`_optimize_hrp`, `optimizers.py:212-239`): single-linkage hardcoded (not configurable), no Ledoit-Wolf shrinkage, covariance is plain annualized `returns.cov() * frequency` — a **separate code path from `hrp.py`'s `HRPOptimizer`**, not a call into it. Two independent HRP implementations coexist in this codebase.
- Every method's raw weights get passed through `constraints.project_to_bounds()` afterward (`optimizers.py:102-107`) — necessary because HRP and equal-weight never look at `constraints` at all, so without that step they'd silently ignore Position Limits/turnover entirely.
- `efficient_frontier()` (`optimizers.py:380-465`) explicitly uses `maxiter=1000` (matching `_run_optimization`'s value) because "the default of 100 is too low for frontier points which carry an extra equality constraint" (`optimizers.py:429-432`); accepts near-converged points via `result.success or (result.fun > 0 and weights_feasible)` rather than `result.success` alone, and recomputes volatility directly from weights rather than trusting `result.fun` (floating-point artifacts).

### `hrp.py` (275 lines) — `HRPOptimizer`, the "real" HRP

- Correlation: plain pandas `.corr()`. Covariance: `sklearn.covariance.LedoitWolf` shrinkage **by default** (`use_shrinkage=True`), falling back to plain `.cov()` only if disabled.
- Classic López de Prado 3-step algorithm: tree clustering via `scipy.cluster.hierarchy.linkage` (default `linkage_method='single'`) → quasi-diagonalization via `leaves_list` → recursive bisection with inverse-variance cluster allocation. Distance metric `d = sqrt(0.5*(1-rho))` (note: a different constant, 0.5 not 2.0, from `src/portfolio_builder/network.py`'s Mantegna `d = sqrt(2*(1-rho))` — these are two different, both-standard MST/clustering distance conventions used in different parts of the codebase, not an inconsistency to "fix").
- `get_dendrogram_data()` and `get_cluster_members()` each independently recompute the distance/linkage matrices rather than reusing `optimize()`'s own output — minor internal duplication, not a correctness issue.

### `black_litterman.py` (445 lines) — `BlackLittermanModel`

- Market-cap-implied equilibrium returns via reverse optimization (`pi = delta * Sigma * w_mkt`, `risk_aversion` default 2.5), falling back to equal weights if no `market_caps` given. Investor views (`add_absolute_view`/`add_relative_view`), each with a `confidence` in `[0.01, 0.99]` converted to view uncertainty `omega = view_var * (1/confidence - 1)` — higher confidence means lower uncertainty.
- Standard Bayesian posterior-returns formula, then a max-Sharpe SLSQP solve over the posterior, with `constraints.compute_bounds()` for turnover-aware bounds — but **unlike `optimizers.py`, does NOT call `project_to_bounds()` afterward**. This is consistent, not an oversight: BL's own solve is already bounds-respecting (SLSQP-bounded), so it doesn't need the "bounds-blind method" safety net that HRP/equal-weight require.
- Plain `.cov()` for the covariance matrix here — no Ledoit-Wolf shrinkage (unlike `hrp.py`'s default).

### `constraints.py` (561 lines) — `PortfolioConstraints`, shared by all three backends above

Three **unrelated** mechanisms in this codebase are all called "turnover" — do not conflate them in documentation or in future changes:

1. **`PortfolioConstraints.turnover_enabled` / `reduction_pct` (0.50) / `increase_pct` (0.30) / `allow_full_exit` (True)** — a real, tested, actively-used per-ticker weight trading-band: each currently-held ticker's bounds become `[current_w*(1-reduction_pct), current_w*(1+increase_pct)]` (clamped to `min_weight`/`max_weight`), applied via `compute_bounds()`. Regression-tested in `tests/test_turnover_constraint.py` across both `PortfolioOptimizer` and `BlackLittermanModel` — the test file's own docstring notes Black-Litterman previously ignored this band entirely, and the tests guard against that regressing again.
2. **`PortfolioConstraints.max_turnover` / `get_turnover_constraint()`** — an older, simpler L1-total-turnover cap (`max_turnover - sum(abs(weights - current_weights)) >= 0`, built as a scipy constraint dict). **Appears unused/dead**: defined but not called anywhere in `optimizers.py`, `black_litterman.py`, or elsewhere in the codebase per grep — the actively-used mechanism is #1 above.
3. **`src.portfolio_builder.ranking.TurnoverConfig.entry_percentile`/`stay_percentile`** — a completely different, currently-unset placeholder governing which *stocks* enter/stay in the ranked universe (a selection-stage concept), not a weight constraint at all. See the Portfolio Builder section above and CLAUDE.md's Known Placeholders.

Other constraints supported: min/max weight, min/max position size, sector caps (`get_sector_constraints()`), `long_only`. Volatility/return targets (`target_volatility`, `max_volatility`, `min_return`, `max_drawdown`) are checked post-hoc in `check_constraints()` only — never passed into any scipy solve as a hard constraint.

- `compute_bounds()` (turnover-aware, actively used) vs. `get_bounds()` (simpler flat bounds, used only by `efficient_frontier()`, not turnover-aware) — two different bounds functions, easy to mix up by name.
- `project_to_bounds()`: iterative water-filling — clip to bounds, redistribute the remaining weight budget proportionally among still-active assets, repeat until every position is either pinned or the budget is exhausted.
- `apply_minimum_position()`: repeat-until-stable renormalization loop; if every position would be eliminated by the minimum-size threshold, falls back to keeping only the top-k largest.

<a id="risk"></a>
## `src/risk/` — volatility, correlation, VaR, stress inputs

### ⚠️ Correction to the Portfolio Builder's own stated rationale — read this first

`src/portfolio_builder/network.py` and `fetch.py` justify staying at plain ticker-level `.corr()` (instead of extending `DCCGARCHModel` to ticker level) by citing "an unresolved convergence-misreport issue" in the DCC-GARCH engine. **Independent audit of `src/risk/dcc_garch.py` itself found no comment, docstring, or code path matching that description.** What actually exists is a hard guard, not a silent misreport:

```python
# dcc_garch.py:418-425
n_failed = sum(1 for v in convergence_status.values() if not v)
if n_failed > N / 2:
    raise ConvergenceError(
        f"{n_failed}/{N} GARCH models failed to converge. "
        "Try: increase max_fit_attempts, use mean_model='zero', "
        "or distribution='normal'."
    )
```

with an explicit constant-volatility fallback for any single series that fails (`dcc_garch.py:407-409`, `# Total failure — constant-vol fallback`) — this is proper failure handling, not an unresolved bug. Two readings are possible: (a) the portfolio_builder docstrings are stale relative to a since-fixed `dcc_garch.py`, or (b) the concern is about the underlying `arch` package's own per-series convergence flag being unreliable in ways `convergence_status` wouldn't catch — which, if true, isn't documented anywhere inside `dcc_garch.py` itself. Either way: **the decision to stay at ticker-level plain correlation is still fine as a decision** (there's no need for ticker-level DCC-GARCH), but the specific justification quoted in `network.py`/`fetch.py`'s own docstrings does not currently trace to anything verifiable inside `dcc_garch.py`. Flag this for whoever next revisits that decision, rather than repeating the claim as settled fact.

### `dcc_garch.py` (769 lines) — `DCCGARCHModel`

Engle (2002) DCC-GARCH: fits a univariate GARCH(p,q) per **sector** via the `arch` package, then runs a DCC recursion on standardized residuals. **Confirmed sector-level, not ticker-level** — `fit(sector_returns: pd.DataFrame)` expects sector-named columns; called from `src/simulation/sector_stress.py` and `app/pages/10_Portfolio_Builder.py` with sector-aggregated returns (production N ≈ 11, the TRBC/GICS sector count). `get_correlation_at_quantile()` picks a single historical timestep's correlation matrix at a given percentile of average conditional volatility (default `vol_stress_quantile=0.95`) — a quantile *selection* over the already-fitted series, not a re-fit on a shorter window. `forecast_correlation()` is a mean-reverting projection (`Q_{T+h} = (1-(alpha+beta)^h)*Q_bar + (alpha+beta)^h*Q_T`) — this is the function `src/portfolio_builder/metrics.py` explicitly refuses to call, at any horizon (see that section above).

### `garch.py` (495 lines) — `GARCHModel` / `MultiAssetGARCH`

Single-asset volatility only (no cross-asset correlation — that's `dcc_garch.py`'s job). Supports GARCH(1,1) (default), EGARCH, and GJR-GARCH via the `arch` package, with a simplified closed-form fallback if `arch` isn't installed. `MultiAssetGARCH` fits one *independent* `GARCHModel` per column — still no correlation modeling across assets.

### `copula.py` (764 lines) — `StudentTCopula`

Student-t (default) or Gaussian copula fit on **sector** returns, for tail dependence beyond linear beta and conditional joint-scenario simulation (`simulate_conditional()`). `_nearest_positive_definite()` uses eigendecomposition (Higham 2002) to repair a near-singular correlation matrix.

### `contagion.py` (845 lines) — `LeontifContagionEngine`

Leontief input-output contagion for the IDX macro stress model: propagates initial sector distress `h(0)` through an inter-sector weight matrix `W` with nonlinear saturation (`h(t+1) = h(0) + (W·h(t))*(1-h(t))` — a sector at 90% distress absorbs at most 10% more incoming contagion, by construction) until convergence, with an optional IDR feedback loop. `get_cascade_risk_label()` classifies spectral radius against `cascade_warning_threshold=0.90` / `cascade_critical_threshold=0.98`.

### `regime_detection.py` (686 lines) — `MarketRegimeDetector`

Gaussian **HMM** (`hmmlearn.hmm.GaussianHMM`, not a Markov-switching regression) fit on engineered sector-return features (rolling volatility, mean return). States are relabeled post-fit by mean conditional volatility so state 0 is always "calm" and the highest index is always "crisis" (2-4 state label sets, e.g. `{0: calm, 1: crisis}` up to `{0: calm, 1: mild_stress, 2: elevated, 3: crisis}`).

**This is the actual "crisis-mode correlation" mechanism in this codebase** (there is no literal `min_overlap_days` concept anywhere in `src/risk/` — confirmed by exhaustive grep, do not document one as existing): `get_regime_correlation(regime_label, dcc_result, regime_result)` averages `DCCGARCHModel`'s fitted correlation matrices over every historical date classified into a given regime (e.g. every date the HMM labeled "crisis"). If fewer than 5 aligned observations exist for that regime, it falls back to an **identity matrix** (`regime_detection.py:459-464`), not a shorter/alternate window — a genuine minimum-sample guard, not flagged anywhere in the code as a placeholder or unvalidated constant.

### `var.py` (559 lines) — `VaRCalculator` / `PortfolioVaR`

Historical, parametric (variance-covariance), and Cornish-Fisher VaR; historical and parametric CVaR/Expected Shortfall; rolling VaR; VaR backtesting. **No Monte Carlo VaR method exists in this file** — that lives in `src/simulation/monte_carlo.py`'s `value_at_risk()`/`conditional_var()` instead, a separate implementation.

### `metrics.py` (673 lines) — `RiskMetrics`, a SEPARATE Sharpe implementation from `src/portfolio_builder/metrics.py`

Broad general-purpose risk/performance metrics: volatility family, Sharpe/Sortino/Calmar/Omega/Information/Treynor ratios, M², drawdown family, higher moments, diversification measures. **Confirmed: this file's `sharpe_ratio()` has no `lookback_days` or `SharpeConfig` concept at all** (zero grep matches) — it's a plain full-series calculation against a scalar `risk_free_rate`/`frequency` set at construction. This is a completely independent implementation from `src/portfolio_builder/metrics.py`'s `SharpeConfig`/`compute_sharpe()` (3-year lookback, DCC-GARCH-trailing-average volatility) — **confirmed zero cross-references either direction** by grep. Two unrelated "Sharpe ratio" implementations coexist in this codebase; know which one a given page/module is using.

### `sector_beta.py` (1008 lines) vs. `stock_sector_beta.py` (484 lines)

- `sector_beta.py`: `SectorBetaAnalyzer` — primarily **sector-to-sector** beta (dual short/long window, 252/756 days, with a stability flag when the two windows disagree). Also contains two secondary per-stock methods (`compute_stock_to_sector_betas`, `compute_stock_betas_vs_portfolio_sectors`) that per grep have **zero external call sites** — unused by the rest of the codebase.
- `stock_sector_beta.py`: `compute_all_stock_betas()` — the actively-used per-stock path, with ETF-ex-stock circularity correction for dominant holdings (`CIRCULARITY_THRESHOLD=0.10`, hardcoded `KNOWN_DOMINANT_WEIGHTS` for e.g. NVDA/AAPL/MSFT-in-XLK).
- **Stale-docstring discrepancy, worth fixing or at least flagging**: `sector_beta.py`'s `compute_stock_betas_vs_portfolio_sectors()` docstring claims to be "the primary stock-beta path used by `SectorStressEngine.fit()`" — this is **inaccurate**. `src/simulation/sector_stress.py`'s `fit()` actually imports and calls `compute_all_stock_betas` from `stock_sector_beta.py` instead (with its own comment: "Per-stock sector-relative betas (ETF OLS with circularity fix)"). Don't repeat the stale claim in any documentation; the real production per-stock beta path is in `stock_sector_beta.py`.

### `macro_sensitivity.py` (643 lines) — `MacroSensitivityEstimator`

Estimates sector×macro-factor sensitivity matrix `S` (`r_i = alpha_i + S[i,:]·X + eps_i`) via OLS or Ridge (default, `ridge_alpha=0.01`). Macro factors aren't defined here — supplied by the caller as a `macro_data: pd.DataFrame`, documented to come from `src.data.macro_data.MacroDataResult.aligned_weekly` (a data-contract relationship, not a direct import — no `from src.data.macro_data import` statement exists in this file).

### Package-level notes

`src/risk/__init__.py` only re-exports the three "classic" modules: `RiskMetrics`/`calculate_metrics` (`metrics.py`), `VaRCalculator`/`PortfolioVaR` (`var.py`), `GARCHModel`/`MultiAssetGARCH` (`garch.py`). The six sector-level/systemic-risk modules (`dcc_garch.py`, `copula.py`, `contagion.py`, `regime_detection.py`, `sector_beta.py`, `stock_sector_beta.py`, `macro_sensitivity.py`) are not part of the package `__init__` and must be imported by full submodule path. No TODO/FIXME/placeholder markers exist anywhere in this directory (confirmed by exhaustive grep). Most of these modules (`dcc_garch.py`, `copula.py`, `contagion.py`, `regime_detection.py`, `sector_beta.py`, `macro_sensitivity.py`) ship their own `if __name__ == "__main__"` smoke test rather than a `pytest` file; `stock_sector_beta.py` is the exception, covered by `tests/test_stock_sector_beta.py`.

<a id="simulation"></a>
## `src/simulation/` — Monte Carlo + stress scenarios

### `monte_carlo.py` (547 lines) — `MonteCarloSimulator`

Simulates forward paths of **portfolio value** (a single collapsed mean/vol number, not per-asset paths — `mean_returns`/`cov_matrix` are reduced to scalar `portfolio_mean`/`portfolio_vol` once at construction, reused by all 4 methods). Methods: `gbm` (classic geometric Brownian motion), `bootstrap` (block bootstrap of actual historical portfolio returns, preserves short-term autocorrelation, non-parametric), `student_t` (fat-tailed; degrees of freedom estimated from sample excess kurtosis if not given), `jump_diffusion` (GBM + compound-Poisson jump term). `run_multi_horizon()` defaults to `[21, 63, 126, 252, 756]` days (1m/3m/6m/1y/3y).

### `scenarios.py` (564 lines) — legacy uniform-shock stress tester

`StressTester` applies **hardcoded uniform percentage shocks** (not simulation) — a fixed `HISTORICAL_SCENARIOS` dict (8 entries, e.g. `2008_financial_crisis: -50%`) used as a fallback when no actual price data is supplied to `historical_scenario()`. `parametric_stress()` has an explicit inline comment `# Simple implementation - apply uniform shock` — it's intentionally the crude/legacy method. `run_historical_actual()` bridges to the newer per-stock engine below by delegating to `HistoricalStressor.run_all()`.

**Naming collision worth flagging**: this file defines its own `HISTORICAL_SCENARIOS: dict` (line 20) *and* imports `historical_scenarios.py`'s differently-shaped list aliased `NEW_HISTORICAL_SCENARIOS` — two same-named-in-spirit but structurally different (dict vs. list, 8 vs. 7 entries) scenario collections live in this one module's namespace simultaneously. Don't conflate them when documenting or extending either.

### `historical_scenarios.py` (666 lines) — `HistoricalStressor`, actual per-stock replay

Unlike `scenarios.py`'s uniform shock, this module downloads **real per-ticker yfinance prices** for each of 7 pre-loaded crisis windows (COVID-19, 2008 GFC, 1997 Asian Crisis, 2013 Taper Tantrum, 2022 Bear Market, Dot-com Bust, 2018 Q4 Selloff — each tagged with a market index, `^GSPC` or `^JKSE` for IDX-relevant ones) and computes actual realized returns per stock. For stocks lacking sufficient crisis-window data (`< min_data_points`, default 5), falls back to a **beta-scaled index return** (pre-crisis OLS beta, clipped to `[-3.0, 3.0]`), not a flat/uniform shock. `to_comparison_dataframe()` and `to_stock_breakdown()` provide the two output views the UI consumes.

### `macro_stress.py` (871 lines) — `MacroStressEngine`, Leontief contagion orchestration

Fits `MacroDataFetcher` (macro time series) → `SectorBetaAnalyzer.build_sector_returns()` (aggregate to sector returns) → `MacroSensitivityEstimator.estimate()` (OLS/Ridge sensitivity matrix **S**, sectors × macro factors) → at stress time, `S · macro_shock` gives initial per-sector distress, fed into `LeontiefContagionEngine.propagate()`'s iterative nonlinear-saturation contagion model (with an IDR-exchange-rate feedback loop) using a sector-weight matrix **W** built from `SectorBetaAnalyzer`'s beta matrix. 6 default scenarios (`DEFAULT_MACRO_SCENARIOS`), each carrying a `historical_reference` string explicitly documented as "documentation only, not used computationally."

### `sector_stress.py` (1114 lines) — `SectorStressEngine`, the most elaborate of the five

Chains 4 sub-models, each fit in an **isolated try/except so one failing sub-model doesn't abort the others** (explicit design comment, `sector_stress.py:340-341`): `SectorBetaAnalyzer` (sector-to-sector beta + `compute_all_stock_betas` per-stock ETF beta) → `DCCGARCHModel` (dynamic correlation) → `StudentTCopula` (conditional tail simulation) → `MarketRegimeDetector` (regime-conditioned correlation selection). 12 default scenarios (`DEFAULT_SCENARIOS`), several with explicit IDX relevance notes (e.g. "Telco Margin Compression: High IDX relevance: TLKM"; "Utility Rate Risk: High IDX relevance: PGEO, PGAS, BREN"). Per-holding P&L combines `stock_beta * sector_return` (beta-implied) with a copula-conditional-simulation quantile draw; correlation-matrix selection priority is regime-conditioned DCC → DCC stress/calm snapshot → copula correlation → identity-matrix fallback.

**Cross-cutting note**: `src/simulation/__init__.py` only re-exports `monte_carlo.py` and `scenarios.py` symbols — `historical_scenarios.py`, `macro_stress.py`, and `sector_stress.py` are not part of the package's `__all__`; all real call sites import them directly from their submodules (e.g. `from src.simulation.macro_stress import MacroStressEngine` in `app/pages/4_Stress_Testing.py`). No pytest coverage exists for any of these 5 files directly — `historical_scenarios.py`, `sector_stress.py`, and `macro_stress.py` each ship a manual smoke test behind `if __name__ == "__main__":` that must be run directly, not via `pytest`.

<a id="valuation"></a>
## `src/valuation/stock_valuer.py` (1295 lines) — fundamentals scoring + reverse DCF

Three independently-callable stages, orchestrated by `analyze_stocks(tickers, wacc=0.10)`: `magic_formula_screen()` (Greenblatt screen: market cap ≥ $500M, positive EBIT, debt/EBITDA ≤ 3x, ranked by EBIT/EV + ROIC) → `multi_factor_score()` (top 30%, min 3 survivors) → `reverse_dcf()` (score ≥ 65 only).

- `multi_factor_score(ticker) -> dict`: 5 sub-scores summing to 100 — Quality (30: ROIC 12, gross-margin trend 10, FCF/NI 8), Value (25: EV/EBIT 10, FCF yield 8, EV/EBITDA-vs-sector 7), Momentum (20: price-vs-200d-SMA 8, 12-1 month return 12), Growth (15: revenue 3yr CAGR 6, EPS-vs-revenue leverage 5, Sloan accruals 4), Health (10: interest coverage 5, debt/EBITDA 5).
- `data_quality_score = round(n_fetched / n_attempted * 100, 1)` — confirmed exact formula, counting successfully-computed sub-metrics over attempted ones across all 5 factor groups.
- **On total fetch failure, `multi_factor_score()` does NOT raise** — it returns a fully-shaped, structurally "successful-looking" all-zero dict (`total_score=0.0`, every sub-score `0.0`, `data_quality_score=0.0`) with only `warnings=["Data fetch failed: ..."]` distinguishing it from a stock that genuinely scored zero on everything. This is exactly why `src/portfolio_builder/fetch.py`'s `_build_entry()` checks `data_quality_score` explicitly rather than trusting `total_score` alone (see the Portfolio Builder section above) — a caller that only reads `total_score` cannot tell a data outage apart from a legitimately bad stock.
- `reverse_dcf(ticker, wacc=0.10) -> dict`: solves via `scipy.optimize.brentq` (bounds `[-0.50, 3.0]`) for the FCF growth rate `g` implied by current market cap, against a hardcoded terminal growth rate `TGR=0.025` (long-run GDP proxy). `growth_premium = implied_growth_rate - historical_fcf_cagr`, with **three branches**, not the simple two originally assumed:
  1. `historical_fcf_cagr` is a real positive number → `growth_premium` computed normally; verdict via *ratio* thresholds (`implied_g <= hist_cagr*1.5` → "Reasonable", `<= *2.0` → "Stretched", else "Extreme").
  2. `historical_fcf_cagr` is a real but **negative** number → `growth_premium` is *still computed* (not `None`) as `implied_g - hist_cagr`; verdict switches to *absolute* thresholds (`<=0.05`/`<=0.15`/else).
  3. `historical_fcf_cagr` is `None` (fewer than 4 annual FCF data points, or too little usable history) → `growth_premium` stays `None` (never fabricated); verdict via a third set of absolute thresholds (`<=0.10`/`<=0.20`/else).
  So `growth_premium is None` only in branch 3 — a computable-but-negative historical CAGR (branch 2) still produces a real `growth_premium` number. `src/portfolio_builder/fetch.py::compute_dcf_gap()`'s "returns `None` when `growth_premium` is `None`" description is accurate to branch 3 specifically.
- If market cap falls outside the DCF's achievable PV range, `implied_growth_rate` is pinned to the nearest search boundary (not `None`, not an exception) with a warning noting it didn't converge.
- Verdict labels (`reverse_dcf`): "Reasonable" / "Stretched" / "Extreme" / "Insufficient Data" (the last is the initialized default, persists if the function returns early — e.g. `wacc <= TGR`, no market cap, non-positive FCF, or brentq failure). These are a **separate labeling scheme** from `analyze_stocks`'s own `"rating"` column ("Strong Buy" ≥80 / "Buy" ≥65 / "Hold" ≥50 / "Underweight" ≥35 / "Avoid" <35) — don't conflate the two in documentation.
- No TODO/FIXME/placeholder markers anywhere in this file.

---

<a id="other"></a>
## Everything else (`src/factors/`, `src/reports/`, `src/portfolio/`, `src/utils/`)

Not in scope for this audit pass (the user's request covered `portfolio_builder`, `valuation`, `risk`, `data`, `optimization`, `simulation` only). Class names spot-checked present by grep: `FamaFrenchAnalyzer`, `StyleFactorAnalyzer`, `SectorAttribution`, `BrinsonAttribution`, `FactorRiskDecomposition` (`src/factors/`); `ReportGenerator`, `ReportChartGenerator` (`src/reports/`); `HoldingsTracker`, `PositionCalculator`, `PortfolioRebalancer`, `DCAScheduler`, `PerformanceAttributor` (`src/portfolio/`). See README.md's existing "Factor Analysis", "Report Generation", and "Portfolio Management" sections for descriptions — not re-verified line-by-line here.

`src/utils/preset_manager.py` — pure persistence for saved portfolio presets (tickers/weights/value snapshots), no network dependency by design. `apply_preset_to_state()` sets `tickers`/`weights`/`current_portfolio_weights`/`portfolio_value`/`settings`/`loaded_preset_id` but does **not** set `current_holdings` — the page-level wrapper `_load_preset_and_populate_holdings()` in `app/pages/1_Portfolio_Input.py` derives `current_holdings` share counts from a live price fetch on top of it.
