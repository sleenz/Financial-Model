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


def _infer_market(ticker: str) -> str:
    """"US" | "IDX" — IDX tickers use the .JK (Jakarta) suffix elsewhere in this codebase."""
    return "IDX" if ticker.upper().endswith(".JK") else "US"


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

    mfs = fundamentals.get("multi_factor") or {}
    dcf = fundamentals.get("dcf") or {}

    composite_score = float(mfs.get("total_score", 0.0))
    factor_zscores = {
        # Interim proxies from stock_valuer's existing pipeline — see module
        # docstring. Phase 2's RankingEngine replaces these with true
        # sector-neutral z-scores over (earnings_yield, roc, momentum, dcf_gap).
        "quality_score": mfs.get("quality_score", 0.0),
        "value_score": mfs.get("value_score", 0.0),
        "momentum_score": mfs.get("momentum_score", 0.0),
        "growth_score": mfs.get("growth_score", 0.0),
        "health_score": mfs.get("health_score", 0.0),
        "implied_growth_rate": dcf.get("implied_growth_rate"),
        "growth_premium": dcf.get("growth_premium"),
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

        entry = _build_entry(ticker, self._data_layer)
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
        from src.portfolio_builder.fetch import OnDemandFetcher

        class _MockDataLayer:
            """No network calls — exercises OnDemandFetcher's control flow only."""

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
                        "warnings": [],
                    },
                    "dcf": {"implied_growth_rate": 0.08, "growth_premium": 0.02, "warnings": None},
                    "warnings": [],
                }

        cache = UniverseCache(CacheConfig(cache_path=":memory:"))
        fetcher = OnDemandFetcher(cache, _MockDataLayer())

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
