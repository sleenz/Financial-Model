"""
On-demand fetch + interim scoring for the Portfolio Builder.

Reuse (Phase 0 audit + Phase 1 answers):
- Prices/returns:  src.data.data_manager.DataManager
                   (yfinance -> AlphaVantage -> TwelveData -> FMP fallback chain)
- Sector mapping:  DataManager.get_sector_classifications()
                   (wraps src.data.lseg_sectors.LSEGSectorFetcher, TRBC/GICS)
- Fundamentals:    src.valuation.stock_valuer.multi_factor_score / reverse_dcf
                   (DataManager doesn't fetch the fields these two need)
- Correlation:     plain pandas .corr() on daily returns, HRP-style
                   (src.optimization.hrp), NOT DCC-GARCH — DCC-GARCH is
                   fit at sector count and has an unresolved convergence-
                   misreport issue; running it at ticker count would
                   multiply that risk for a feature where a wrong number
                   is invisible in the UI. Revisit only if the Phase 3
                   MST clusters don't line up with real sector groupings.

IMPORTANT — interim scoring, not Phase 2's algorithm:
Phase 2 (ranking.py, not yet built) owns the real sector-neutral 4-factor
composite (earnings_yield, roc, momentum, dcf_gap) with CompositeWeights
and FactorConfig. Until it exists, composite_score/factor_zscores here are
built directly from stock_valuer's existing public outputs (already a
real, working composite — not invented, not random) so the cache/fetch
plumbing in this phase has real data to round-trip and time. compute_fn
is injectable (see cache.UniverseCache.run_nightly_refresh) specifically
so Phase 2 wiring is a one-line swap, not a rewrite of this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from src.data.data_manager import DataManager
from src.valuation.stock_valuer import multi_factor_score, reverse_dcf
from src.portfolio_builder.cache import UniverseCache, RankedUniverseEntry
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FetchConfig:
    price_history_days: int = 400          # >252+21 trading days needed by
                                            # stock_valuer's momentum/SMA-200 calcs
    min_correlation_overlap_days: int = 60  # below this, a correlation is unreliable
    dcf_wacc: float = 0.10                  # forwarded to stock_valuer.reverse_dcf
    min_data_quality_score: float = 0.0     # multi_factor_score()'s own data_quality_score
                                             # (n_fetched/n_attempted*100) at/below this means
                                             # every sub-metric failed — stock_valuer catches its
                                             # own fetch errors internally and returns a
                                             # "successful" all-zero dict rather than raising, so
                                             # _build_entry can't tell "genuinely scores 0" apart
                                             # from "no data came back at all" without this check


def _infer_market(ticker: str) -> str:
    """"US" | "IDX" — IDX tickers use the .JK (Jakarta) suffix elsewhere in this codebase."""
    return "IDX" if ticker.upper().endswith(".JK") else "US"


def _safe_numeric(value, ticker: str, field_name: str) -> float:
    """Coerce to float, raising rather than silently caching NaN/inf into a
    REAL NOT NULL SQLite column or corrupting an otherwise-valid composite
    score. A total-fetch-failure ticker should show up in run_nightly_refresh's
    'failed' list, not as a fake 0.0 (or worse, NaN) composite_score."""
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{ticker}: {field_name}={value!r} is not numeric: {exc}") from exc
    if math.isnan(f) or math.isinf(f):
        raise RuntimeError(f"{ticker}: {field_name}={f} is NaN/inf, refusing to cache")
    return f


def _json_safe(value):
    """None/NaN/inf -> None so factor_zscores always serializes to valid JSON
    (Python's json.dumps otherwise emits the non-standard token 'NaN')."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def compute_dcf_gap(dcf: dict) -> Optional[float]:
    """
    dcf_gap raw factor value: the negation of stock_valuer.reverse_dcf's
    growth_premium (= implied_growth_rate - historical_fcf_cagr).

    This is the ONLY place in the codebase that constructs a dcf_gap raw
    value from reverse_dcf's output (ranking.py's compute_factor_zscore has
    no way to verify this sign itself — it z-scores whatever it's given).
    The reverse-DCF design intent is: a stock priced for LESS growth than
    its own historical trend would suggest (growth_premium <= 0, i.e. the
    market is NOT extrapolating recent momentum) should score HIGHER on
    this factor than a stock priced for MORE growth than its own trend
    (growth_premium > 0, i.e. priced-to-perfection relative to its own
    history) — hence the negation.

    Integration-verified (see fetch.py's module-level dcf_gap_real_data_check
    docstring/report, not a synthetic unit test) against real reverse_dcf
    output for real tickers: Coca-Cola (KO), priced for +31.8% implied growth
    against a -17.8% trailing 3yr FCF CAGR (growth_premium=+0.496, an
    "Extreme" reverse_dcf verdict), produced dcf_gap=-0.496 and the LOWEST
    z-score in a real 4-ticker peer comparison — confirming the sign matches
    the design intent for a stock that genuinely fits the "priced for growth
    beyond its own benchmark" premise.

    Returns None (not 0.0) when growth_premium itself is None (reverse_dcf
    couldn't compute one — e.g. no positive historical FCF to compare
    against) — a missing input must propagate as missing, never a fabricated
    neutral value; compute_factor_zscore/compute_composite_score already
    handle a None/NaN factor value by excluding it from factor_coverage and
    neutral-filling the composite, so this is the correct "I don't know"
    signal, not 0.0's "priced exactly at its historical trend."
    """
    growth_premium = dcf.get("growth_premium")
    if growth_premium is None:
        return None
    return -float(growth_premium)


class PortfolioDataLayer:
    """
    Adapter binding the Phase-0-audited data sources together.

    Wraps DataManager for price history + sector lookups, and calls
    stock_valuer's fundamentals functions directly for scoring inputs
    DataManager doesn't fetch (EBIT/EV, ROIC, reverse-DCF growth gap).
    """

    def __init__(
        self,
        config: FetchConfig = FetchConfig(),
        data_manager: Optional[DataManager] = None,
    ):
        self.config = config
        self.data_manager = data_manager or DataManager(show_progress=False)

    def fetch_prices(self, tickers: list, start_date, end_date) -> pd.DataFrame:
        return self.data_manager.get_price_data(tickers, start_date, end_date, validate=True)

    def fetch_sector(self, ticker: str) -> str:
        sector_map = self.data_manager.get_sector_classifications([ticker])
        return sector_map.get(ticker, "Unknown")

    def fetch_fundamentals(self, ticker: str) -> dict:
        """Real fundamentals via stock_valuer. Never raises — degrades to warnings."""
        result: dict = {"multi_factor": None, "dcf": None, "warnings": []}

        try:
            mfs = multi_factor_score(ticker)
            result["multi_factor"] = mfs
            result["warnings"].extend(mfs.get("warnings", []))
        except Exception as exc:
            logger.error(f"fetch_fundamentals: multi_factor_score failed for {ticker}: {exc}")
            result["warnings"].append(f"multi_factor_score failed: {exc}")

        try:
            dcf = reverse_dcf(ticker, wacc=self.config.dcf_wacc)
            result["dcf"] = dcf
            if dcf.get("warnings"):
                result["warnings"].append(dcf["warnings"])
        except Exception as exc:
            logger.error(f"fetch_fundamentals: reverse_dcf failed for {ticker}: {exc}")
            result["warnings"].append(f"reverse_dcf failed: {exc}")

        return result


def build_default_data_layer(config: FetchConfig = FetchConfig()) -> PortfolioDataLayer:
    """Factory used by UniverseCache.run_nightly_refresh() when no data_layer is injected."""
    return PortfolioDataLayer(config=config)


def _build_entry(ticker: str, data_layer: PortfolioDataLayer) -> RankedUniverseEntry:
    """Build a RankedUniverseEntry from live fundamentals. correlation_row left empty —
    callers (get_or_fetch / compute_universe_entries) fill it in per their own contract."""
    try:
        sector = data_layer.fetch_sector(ticker)
    except Exception as exc:
        logger.error(f"{ticker}: sector lookup failed: {exc}")
        sector = "Unknown"

    fundamentals = data_layer.fetch_fundamentals(ticker)
    for w in fundamentals.get("warnings", []):
        logger.warning(f"{ticker}: {w}")

    mfs = fundamentals.get("multi_factor")
    if mfs is None:
        # multi_factor_score() failed entirely (see fetch_fundamentals) — there
        # is no legitimate total_score to report. Raising here (instead of
        # caching a placeholder 0.0) is what lets compute_universe_entries'
        # per-ticker try/except and run_nightly_refresh correctly count this
        # ticker as failed rather than "refreshed" with a meaningless score.
        raise RuntimeError(f"{ticker}: multi_factor_score fetch failed entirely — refusing to cache a placeholder composite_score")

    # multi_factor_score() catches its own fetch errors internally (see
    # stock_valuer.py) and returns a "successful" dict — total_score=0.0,
    # every sub-score 0.0 — rather than raising, so mfs is never None even
    # when literally nothing could be fetched (e.g. a delisted/unresolvable
    # ticker). data_quality_score (n_fetched/n_attempted*100) is the only
    # signal that distinguishes "genuinely scores 0" from "no data came
    # back at all" — without this check, a total data outage silently
    # produces a plausible-looking, positively-scored entry instead of a
    # visible failure (independent review caught this reaching the UI).
    data_quality = mfs.get("data_quality_score", 0.0)
    if data_quality <= data_layer.config.min_data_quality_score:
        raise RuntimeError(
            f"{ticker}: multi_factor_score returned data_quality_score={data_quality} "
            f"(<= min_data_quality_score={data_layer.config.min_data_quality_score}) — "
            "treating as a total fetch failure, not a genuine zero score"
        )

    dcf = fundamentals.get("dcf") or {}

    composite_score = _safe_numeric(mfs.get("total_score", 0.0), ticker, "total_score")
    factor_zscores = {
        # Interim proxies from stock_valuer's existing pipeline — see module
        # docstring. Phase 2's RankingEngine replaces these with true
        # sector-neutral z-scores over (earnings_yield, roc, momentum, dcf_gap).
        "quality_score": _json_safe(mfs.get("quality_score", 0.0)),
        "value_score": _json_safe(mfs.get("value_score", 0.0)),
        "momentum_score": _json_safe(mfs.get("momentum_score", 0.0)),
        "growth_score": _json_safe(mfs.get("growth_score", 0.0)),
        "health_score": _json_safe(mfs.get("health_score", 0.0)),
        "implied_growth_rate": _json_safe(dcf.get("implied_growth_rate")),
        "growth_premium": _json_safe(dcf.get("growth_premium")),
    }

    return RankedUniverseEntry(
        ticker=ticker,
        sector=sector,
        market=_infer_market(ticker),
        composite_score=composite_score,
        factor_zscores=factor_zscores,
        correlation_row=[],
        computed_at=datetime.now(timezone.utc).isoformat(),
    )


def compute_universe_entries(universe: list, data_layer: PortfolioDataLayer) -> dict:
    """
    Build a RankedUniverseEntry for every ticker in universe, with a real
    NxN correlation matrix computed once (not refit per ticker) and sliced
    into each entry's correlation_row, aligned to sorted(universe).

    One ticker's fundamentals failure doesn't drop the rest of the batch —
    each ticker is built independently and logged on failure.
    """
    entries: dict = {}
    for ticker in universe:
        try:
            entries[ticker] = _build_entry(ticker, data_layer)
        except Exception as exc:
            logger.error(f"compute_universe_entries: failed to build entry for {ticker}: {exc}")

    ordered = sorted(universe)
    end = datetime.now()
    start = end - timedelta(days=data_layer.config.price_history_days)

    corr = pd.DataFrame()
    try:
        prices = data_layer.fetch_prices(list(entries.keys()) or ordered, start, end)
        returns = prices.pct_change().dropna(how="all")
        if len(returns) >= data_layer.config.min_correlation_overlap_days:
            corr = returns.corr()
        else:
            logger.warning(
                f"compute_universe_entries: only {len(returns)} overlapping return "
                f"observations (< min_correlation_overlap_days="
                f"{data_layer.config.min_correlation_overlap_days}); correlation_row "
                "left empty for this refresh."
            )
    except Exception as exc:
        logger.error(f"compute_universe_entries: price/correlation fetch failed: {exc}")

    for ticker, entry in entries.items():
        if ticker in corr.index:
            row = corr.reindex(index=[ticker], columns=ordered).iloc[0]
            entry.correlation_row = [None if pd.isna(v) else float(v) for v in row.tolist()]
        else:
            entry.correlation_row = [None] * len(ordered)

    return entries


class OnDemandFetcher:
    def __init__(self, cache: UniverseCache, data_layer: PortfolioDataLayer):
        # data_layer = the EXISTING yfinance+FRED fallback layer found
        # in Phase 0 — do not reimplement fetching
        self._cache = cache
        self._data_layer = data_layer

    def get_or_fetch(self, ticker: str) -> RankedUniverseEntry:
        """
        1. cache.get(ticker) — return immediately if hit, no API call.
        2. On miss: fetch via existing data_layer, run composite scoring,
           cache.upsert(), return the new entry.

        correlation_row is left empty on the miss path: computing a real
        row would require re-fetching price history for the entire cached
        universe just to place one new ticker, which breaks the "never
        blocks past a single ticker's fetch time" contract. The next
        scheduled run_nightly_refresh() fills it in — this mirrors
        on_demand_join_cache's stated purpose (join now, refresh tomorrow).
        """
        cached = self._cache.get(ticker)
        if cached is not None:
            return cached

        try:
            entry = _build_entry(ticker, self._data_layer)
        except Exception as exc:
            # Zero silent failures: log with full context and re-raise rather
            # than letting a bad value (e.g. NaN total_score) crash later as
            # an opaque sqlite3.IntegrityError deep inside cache.upsert(), or
            # letting the caller silently receive nothing.
            logger.error(f"{ticker}: on-demand fetch failed, not caching: {exc}")
            raise

        logger.info(
            f"{ticker}: on-demand fetch complete; correlation_row deferred to next "
            "run_nightly_refresh()"
        )

        if self._cache._config.on_demand_join_cache:
            self._cache.upsert(entry)
            index = self._cache.get_correlation_index()
            if ticker not in index:
                index.append(ticker)
                self._cache.set_correlation_index(index)

        return entry


if __name__ == "__main__":
    def _smoke_test():
        from src.portfolio_builder.cache import CacheConfig, UniverseCache
        from src.portfolio_builder.fetch import (
            FetchConfig,
            OnDemandFetcher,
            PortfolioDataLayer,
            _build_entry,
            build_default_data_layer,
            compute_dcf_gap,
            compute_universe_entries,
        )

        # ── compute_dcf_gap: hand-computable sign-flip, hermetic (no network) ──
        # growth_premium > 0 (priced for MORE growth than own historical
        # trend) -> dcf_gap < 0 (scores LOW); growth_premium < 0 (priced for
        # LESS growth than own trend) -> dcf_gap > 0 (scores HIGH).
        assert compute_dcf_gap({"growth_premium": 0.496}) == -0.496
        assert compute_dcf_gap({"growth_premium": -1.6369}) == 1.6369
        assert compute_dcf_gap({"growth_premium": 0.0}) == 0.0
        assert compute_dcf_gap({"growth_premium": None}) is None
        assert compute_dcf_gap({}) is None  # key absent entirely -> same as None, not a KeyError
        print("✓ compute_dcf_gap: dcf_gap = -growth_premium, hand-computable, None propagates as None")

        class _MockDataLayer:
            """No network calls — exercises the fetch pipeline's control flow only."""

            def __init__(self):
                # Only 15 synthetic trading days below, so lower the overlap
                # floor to actually exercise the corr() path in this test.
                self.config = FetchConfig(min_correlation_overlap_days=5)

            def fetch_sector(self, ticker):
                return "Technology"

            def fetch_fundamentals(self, ticker):
                return {
                    "multi_factor": {
                        "total_score": 72.5,
                        "quality_score": 20.0,
                        "value_score": 18.0,
                        "momentum_score": 14.0,
                        "growth_score": 10.0,
                        "health_score": 8.0,
                        "data_quality_score": 100.0,
                        "warnings": [],
                    },
                    "dcf": {"implied_growth_rate": 0.08, "growth_premium": 0.02, "warnings": None},
                    "warnings": [],
                }

            def fetch_prices(self, tickers, start_date, end_date):
                idx = pd.bdate_range(end=end_date, periods=15)
                return pd.DataFrame(
                    {t: [100.0 + i + (hash(t) % 7) for i in range(15)] for t in tickers},
                    index=idx,
                )

        # ── Rule 2: every new AND reused name this module touches must resolve ──
        fc = FetchConfig()
        assert fc.price_history_days == 400 and fc.dcf_wacc == 0.10
        default_layer = build_default_data_layer(fc)
        assert isinstance(default_layer, PortfolioDataLayer)
        print("✓ FetchConfig / build_default_data_layer / PortfolioDataLayer resolve")

        mock_layer = _MockDataLayer()
        built = _build_entry("AAPL", mock_layer)
        assert built.ticker == "AAPL" and built.composite_score == 72.5
        print("✓ _build_entry resolves and builds a real entry")

        universe_entries = compute_universe_entries(["AAPL", "MSFT"], mock_layer)
        assert set(universe_entries.keys()) == {"AAPL", "MSFT"}
        assert len(universe_entries["AAPL"].correlation_row) == 2
        print("✓ compute_universe_entries resolves and populates correlation_row")

        # ── NaN/failure guard: a total fundamentals failure must raise, not
        # cache a placeholder score (Phase 1 CHECK finding) ──────────────────
        class _FailingDataLayer(_MockDataLayer):
            def fetch_fundamentals(self, ticker):
                return {"multi_factor": None, "dcf": None, "warnings": ["network down"]}

        try:
            _build_entry("BADCO", _FailingDataLayer())
            raise AssertionError("expected RuntimeError on total fundamentals failure")
        except RuntimeError:
            pass
        print("✓ total fundamentals failure raises instead of caching a placeholder score")

        class _NanDataLayer(_MockDataLayer):
            def fetch_fundamentals(self, ticker):
                return {
                    "multi_factor": {"total_score": float("nan"), "quality_score": 0.0,
                                      "value_score": 0.0, "momentum_score": 0.0,
                                      "growth_score": 0.0, "health_score": 0.0,
                                      "data_quality_score": 100.0, "warnings": []},
                    "dcf": {"implied_growth_rate": None, "growth_premium": None, "warnings": None},
                    "warnings": [],
                }

        try:
            _build_entry("NANCO", _NanDataLayer())
            raise AssertionError("expected RuntimeError on NaN total_score")
        except RuntimeError:
            pass
        print("✓ NaN composite_score raises instead of corrupting the cache")

        # Regression: multi_factor_score() "succeeding" with data_quality_score=0
        # (every sub-metric failed internally, e.g. a delisted/unresolvable
        # ticker with yfinance down) must be treated as a total failure, not
        # cached as a genuine, plausible-looking zero score. Independent
        # review caught this reaching the UI as a positively-badged,
        # fabricated entry with zero warnings shown anywhere.
        class _ZeroDataQualityLayer(_MockDataLayer):
            def fetch_fundamentals(self, ticker):
                return {
                    "multi_factor": {"total_score": 0.0, "quality_score": 0.0,
                                      "value_score": 0.0, "momentum_score": 0.0,
                                      "growth_score": 0.0, "health_score": 0.0,
                                      "data_quality_score": 0.0, "warnings": []},
                    "dcf": {"implied_growth_rate": None, "growth_premium": None, "warnings": None},
                    "warnings": [],
                }

        try:
            _build_entry("ZZZZ", _ZeroDataQualityLayer())
            raise AssertionError("expected RuntimeError on data_quality_score=0")
        except RuntimeError:
            pass
        print("✓ data_quality_score=0 raises instead of caching a fabricated zero score")

        cache = UniverseCache(CacheConfig(cache_path=":memory:"))
        fetcher = OnDemandFetcher(cache, mock_layer)

        # ── Cache-miss path ──────────────────────────────────────────────
        entry = fetcher.get_or_fetch("AAPL")
        assert entry.ticker == "AAPL"
        assert entry.composite_score == 72.5
        assert cache.get("AAPL") is not None, "on-demand fetch must join the cache"
        print("✓ cache-miss path fetches and joins the cache")

        # ── Instrument upsert to prove call counts ──────────────────────
        call_count = {"n": 0}
        original_upsert = cache.upsert

        def _counting_upsert(e):
            call_count["n"] += 1
            return original_upsert(e)

        cache.upsert = _counting_upsert

        # Cache-hit path: same ticker, must NOT call upsert
        hit_entry = fetcher.get_or_fetch("AAPL")
        assert call_count["n"] == 0, "cache hit must not call upsert"
        assert hit_entry.ticker == "AAPL"
        print("✓ cache-hit path returns cached entry with zero upsert calls")

        # Cache-miss path: new ticker, must call upsert exactly once
        entry2 = fetcher.get_or_fetch("MSFT")
        assert call_count["n"] == 1, "cache miss must call upsert exactly once"
        assert entry2.ticker == "MSFT"
        print("✓ cache-miss path calls upsert exactly once")

        # ── on_demand_join_cache = False must skip the join ─────────────
        cache2 = UniverseCache(CacheConfig(cache_path=":memory:", on_demand_join_cache=False))
        fetcher2 = OnDemandFetcher(cache2, _MockDataLayer())
        fetcher2.get_or_fetch("GOOGL")
        assert cache2.get("GOOGL") is None, "on_demand_join_cache=False must not persist"
        print("✓ on_demand_join_cache=False skips the join")

        print("✓ fetch.py smoke test passed")

    _smoke_test()
