"""
SQLite-backed cache for the Portfolio Builder's ranked universe.

Stores one row per ticker: composite score, factor diagnostics, and this
ticker's row in the cached correlation matrix. This module is a pure
persistence layer — it does not compute scores or correlations itself.
Computation lives in ``fetch.py`` and is injected into
``run_nightly_refresh()`` (default constructed lazily to avoid a circular
import), so this module has no dependency on the scoring pipeline and can
be smoke-tested in total isolation.

Reuse note (Phase 0 audit): this is a new, separate cache from the
existing file/pickle ``src/data/cache.py::DataCache`` used elsewhere in
the app. Different path, different class, different schema — no
collision, but flagged so the two aren't confused during review.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheConfig:
    cache_path: str = "data/portfolio_builder_cache.db"  # SQLite
    cache_ttl_hours: int = 24
    on_demand_join_cache: bool = True
    # uncached tickers join the cache after first fetch, so tomorrow's
    # nightly job picks them up automatically


@dataclass
class RankedUniverseEntry:
    ticker: str
    sector: str
    market: str                # "US" | "IDX"
    composite_score: float
    factor_zscores: dict       # {"earnings_yield": ..., "roc": ...,
                               #  "momentum": ..., "dcf_gap": ...}
    correlation_row: list      # this ticker's row in the cached
                               # correlation matrix, aligned to
                               # a stored ticker index
    computed_at: str           # ISO timestamp — used for TTL check


class UniverseCache:
    """SQLite persistence for RankedUniverseEntry rows, keyed by ticker."""

    def __init__(self, config: CacheConfig = CacheConfig()):
        self._config = config
        # check_same_thread=False + a single shared RLock serializing every
        # method below: Streamlit will share one UniverseCache across
        # request threads (nightly-refresh writer, many concurrent
        # readers). Without the lock, concurrent access to one sqlite3
        # connection from multiple threads corrupts driver-level state
        # (observed: "no more rows available", "cannot commit - no
        # transaction is active", even a raw SystemError from the C
        # extension) — check_same_thread=False alone does not make that
        # safe, it only lifts sqlite3's own same-thread guard. RLock (not
        # a plain Lock) because run_nightly_refresh() re-enters via its
        # own call to self.upsert()/self.set_correlation_index() while
        # already holding the lock on the same thread.
        if config.cache_path != ":memory:":
            Path(config.cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(config.cache_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ranked_universe (
                    ticker           TEXT PRIMARY KEY,
                    sector           TEXT NOT NULL,
                    market           TEXT NOT NULL,
                    composite_score  REAL NOT NULL,
                    factor_zscores   TEXT NOT NULL,
                    correlation_row  TEXT NOT NULL,
                    computed_at      TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Correlation column alignment
    # ------------------------------------------------------------------

    def get_correlation_index(self) -> list:
        """Return the canonical ticker ordering that correlation_row values are aligned to."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM cache_meta WHERE key = 'correlation_index'"
            ).fetchone()
            return json.loads(row["value"]) if row else []

    def set_correlation_index(self, tickers: list) -> None:
        """Replace the canonical ticker ordering. Called by run_nightly_refresh()."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cache_meta (key, value) VALUES ('correlation_index', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (json.dumps(list(tickers)),),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, ticker: str) -> Optional[RankedUniverseEntry]:
        """Return cached entry if present and within cache_ttl_hours.
        None otherwise — caller triggers the on-demand fetch path."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM ranked_universe WHERE ticker = ?", (ticker,)
            ).fetchone()
            if row is None:
                return None

            try:
                computed_at = datetime.fromisoformat(row["computed_at"])
                if computed_at.tzinfo is None:
                    computed_at = computed_at.replace(tzinfo=timezone.utc)
                age_hours = (datetime.now(timezone.utc) - computed_at).total_seconds() / 3600.0
            except (ValueError, TypeError) as exc:
                logger.error(f"UniverseCache.get({ticker}): malformed computed_at, treating as miss: {exc}")
                return None

            if age_hours > self._config.cache_ttl_hours:
                logger.debug(f"UniverseCache: {ticker} expired ({age_hours:.1f}h old)")
                return None

            try:
                return RankedUniverseEntry(
                    ticker=row["ticker"],
                    sector=row["sector"],
                    market=row["market"],
                    composite_score=row["composite_score"],
                    factor_zscores=json.loads(row["factor_zscores"]),
                    correlation_row=json.loads(row["correlation_row"]),
                    computed_at=row["computed_at"],
                )
            except json.JSONDecodeError as exc:
                logger.error(f"UniverseCache.get({ticker}): corrupted JSON columns, treating as miss: {exc}")
                return None

    def upsert(self, entry: RankedUniverseEntry) -> None:
        """Insert or overwrite. Nightly job and on-demand fetch both
        call this — same code path, no special-cased 'first time' logic."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ranked_universe
                    (ticker, sector, market, composite_score, factor_zscores, correlation_row, computed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    sector          = excluded.sector,
                    market          = excluded.market,
                    composite_score = excluded.composite_score,
                    factor_zscores  = excluded.factor_zscores,
                    correlation_row = excluded.correlation_row,
                    computed_at     = excluded.computed_at
                """,
                (
                    entry.ticker,
                    entry.sector,
                    entry.market,
                    float(entry.composite_score),
                    json.dumps(entry.factor_zscores),
                    json.dumps(entry.correlation_row),
                    entry.computed_at,
                ),
            )
            self._conn.commit()

    def run_nightly_refresh(
        self,
        universe: list,
        data_layer: Optional[object] = None,
        compute_fn: Optional[Callable[[list, object], dict]] = None,
    ) -> dict:
        """Recompute composite_score + correlation_row for every ticker
        in universe, upsert all. Called by the scheduled batch job.
        Returns {"refreshed": n, "failed": [...], "duration_s": float}.

        data_layer/compute_fn default to the fetch.py scoring pipeline,
        imported lazily here (not at module load) to avoid a circular
        import between cache.py and fetch.py. Both are injectable so
        tests and future callers can swap in a different scorer without
        changing this method's body.
        """
        t0 = time.time()
        if compute_fn is None or data_layer is None:
            from src.portfolio_builder.fetch import (
                build_default_data_layer,
                compute_universe_entries,
            )
            if data_layer is None:
                data_layer = build_default_data_layer()
            if compute_fn is None:
                compute_fn = compute_universe_entries

        # compute_fn runs outside the lock (it's pure computation + network
        # I/O, no shared DB access) — only the upsert/index-write phase
        # below needs to be serialized against other threads.
        try:
            entries_by_ticker = compute_fn(universe, data_layer)
        except Exception as exc:
            logger.error(f"run_nightly_refresh: batch compute failed entirely: {exc}")
            return {"refreshed": 0, "failed": list(universe), "duration_s": time.time() - t0}

        refreshed = 0
        failed: list = []
        with self._lock:
            for ticker in universe:
                entry = entries_by_ticker.get(ticker)
                if entry is None:
                    failed.append(ticker)
                    continue
                try:
                    self.upsert(entry)
                    refreshed += 1
                except Exception as exc:
                    logger.error(f"run_nightly_refresh: upsert failed for {ticker}: {exc}")
                    failed.append(ticker)

            self.set_correlation_index(sorted(universe))

        duration = time.time() - t0
        logger.info(
            f"run_nightly_refresh: {refreshed}/{len(universe)} refreshed, "
            f"{len(failed)} failed, {duration:.2f}s"
        )
        return {"refreshed": refreshed, "failed": failed, "duration_s": duration}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


if __name__ == "__main__":
    def _smoke_test():
        from src.portfolio_builder.cache import CacheConfig, UniverseCache, RankedUniverseEntry

        cache = UniverseCache(CacheConfig(cache_path=":memory:"))

        # NOTE: the build spec's example used a hardcoded "2026-01-01" literal
        # for computed_at. Since get() enforces cache_ttl_hours against
        # wall-clock now(), a fixed calendar date would eventually go stale
        # and silently flip this smoke test's basic-roundtrip assertion into
        # a TTL-expiry assertion. Using datetime.now() keeps the test
        # evergreen; TTL-expiry itself is exercised explicitly below.
        entry = RankedUniverseEntry(
            "AAPL", "Tech", "US", 0.0, {}, [], datetime.now(timezone.utc).isoformat()
        )
        cache.upsert(entry)
        assert cache.get("AAPL") is not None
        assert cache.get("NONEXISTENT") is None
        print("✓ basic upsert/get roundtrip")

        # TTL expiry
        stale = RankedUniverseEntry(
            "MSFT", "Tech", "US", 50.0, {}, [], "2000-01-01T00:00:00+00:00"
        )
        cache.upsert(stale)
        assert cache.get("MSFT") is None, "stale entry should be TTL-expired"
        print("✓ TTL expiry")

        # Overwrite, not duplicate
        cache.upsert(RankedUniverseEntry(
            "AAPL", "Tech", "US", 99.0, {}, [], datetime.now(timezone.utc).isoformat()
        ))
        row_count = cache._conn.execute(
            "SELECT COUNT(*) c FROM ranked_universe WHERE ticker='AAPL'"
        ).fetchone()["c"]
        assert row_count == 1, "upsert must overwrite, not duplicate"
        assert cache.get("AAPL").composite_score == 99.0
        print("✓ upsert overwrites rather than duplicates")

        # Correlation index round-trip
        cache.set_correlation_index(["AAPL", "MSFT"])
        assert cache.get_correlation_index() == ["AAPL", "MSFT"]
        print("✓ correlation index round-trip")

        print("✓ cache.py smoke test passed")

    _smoke_test()
