"""LSEG TRBC sector classification fetcher with yfinance fallback and TTL caching."""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

import yfinance as yf

from src.data.cache import DataCache
from src.utils.logger import get_logger

try:
    import lseg.data as ld
    _LSEG_AVAILABLE = True
except ImportError:
    ld = None
    _LSEG_AVAILABLE = False

logger = get_logger(__name__)

if not _LSEG_AVAILABLE:
    logger.warning(
        "lseg.data is not installed. LSEG sector lookups will be skipped. "
        "Install with: pip install lseg-data>=2.0.0"
    )


# Bidirectional sector label normalization maps.
# Source of truth: TRBC labels (what LSEG natively returns).
# yfinance fallback returns GICS labels — normalize all to TRBC on ingest.

GICS_TO_TRBC: dict[str, str] = {
    "Consumer Discretionary":  "Consumer Cyclicals",
    "Consumer Staples":        "Consumer Non-Cyclicals",
    "Information Technology":  "Technology",
    "Communication Services":  "Telecommunication Services",
    # These are identical in both systems — listed for completeness:
    "Energy":                  "Energy",
    "Financials":              "Financials",
    "Healthcare":              "Healthcare",
    "Industrials":             "Industrials",
    "Materials":               "Basic Materials",
    "Real Estate":             "Real Estate",
    "Utilities":               "Utilities",
}

TRBC_TO_GICS: dict[str, str] = {v: k for k, v in GICS_TO_TRBC.items()}

# IDX-specific sector overrides.
# Applied AFTER normalization. Use RIC format (with .JK suffix).
# Reason: TRBC assigns based on primary revenue, but some IDX names
# have beta behavior that diverges from their assigned sector.
IDX_SECTOR_OVERRIDES: dict[str, str] = {
    "PGEO.JK":  "Utilities",           # Geothermal — regulated utility behavior
    "GOTO.JK":  "Consumer Cyclicals",  # Super-app — consumer discretionary, not pure tech
    "BREN.JK":  "Utilities",           # Renewable power — utility behavior
    "PGAS.JK":  "Utilities",           # Gas distribution — utility
    "TLKM.JK":  "Telecommunication Services",  # Confirm vs LSEG assignment
    "EMTK.JK":  "Technology",          # Digital finance/tech
}


def normalize_sector_label(
    sector: str,
    source: str,
    target_standard: str = "trbc"
) -> str:
    """
    Normalize a sector label to a consistent standard.

    Parameters
    ----------
    sector : str
        Raw sector label from LSEG or yfinance.
    source : str
        "lseg" or "yfinance". Determines which map to apply.
    target_standard : str
        "trbc" (default) — normalize everything to TRBC labels.
        "gics" — normalize everything to GICS labels.

    Returns
    -------
    str
        Normalized sector label. If not found in map, returns
        original label unchanged (handles unknown sectors).
    """
    if target_standard == "trbc":
        if source == "yfinance":
            return GICS_TO_TRBC.get(sector, sector)
        return sector   # LSEG already returns TRBC
    else:
        if source == "lseg":
            return TRBC_TO_GICS.get(sector, sector)
        return sector   # yfinance already returns GICS


class LSEGConnectionError(RuntimeError):
    """Raised when LSEG Data Library is unavailable and fallback is disabled."""


@dataclass
class LSEGSectorConfig:
    """Configuration for LSEG sector classification fetcher."""

    economic_sector_field: str = field(default="TR.TRBCEconomicSector")
    business_sector_field: str = field(default="TR.TRBCBusinessSector")
    industry_field: str = field(default="TR.TRBCIndustry")
    fallback_to_yfinance: bool = field(default=True)
    yfinance_sector_field: str = field(default="sector")
    cache_ttl_seconds: int = field(default=86400)
    batch_size: int = field(default=50)
    unknown_sector_label: str = field(default="Unknown")
    request_timeout_seconds: int = field(default=30)


@dataclass
class SectorClassification:
    """Sector classification for a single ticker across TRBC hierarchy levels."""

    ticker: str
    economic_sector: str
    business_sector: str
    industry: str
    source: str  # "lseg" | "yfinance" | "unknown"


class LSEGSectorFetcher:
    """
    Fetch TRBC sector classifications from LSEG Data Library with yfinance fallback.

    Handles batching, caching, and graceful degradation when LSEG is unavailable.
    """

    def __init__(self, config: LSEGSectorConfig = LSEGSectorConfig()) -> None:
        """
        Initialise fetcher.

        Parameters
        ----------
        config : LSEGSectorConfig
            Runtime configuration controlling batch size, TTL, fallback behaviour.
        """
        self._config = config
        self._cache = DataCache(ttl_historical=config.cache_ttl_seconds)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_cache_key(self, tickers: list[str]) -> str:
        """Generate a deterministic cache key from a sorted ticker list."""
        key_string = "lseg_sectors|" + "|".join(sorted(tickers))
        return hashlib.md5(key_string.encode()).hexdigest()

    def _fetch_lseg_batch(
        self, batch: list[str]
    ) -> dict[str, SectorClassification]:
        """
        Fetch a single batch of tickers from LSEG Data Library.

        Parameters
        ----------
        batch : list[str]
            Up to config.batch_size RIC strings.

        Returns
        -------
        dict[str, SectorClassification]
            Keyed by ticker.  Missing or null rows are omitted — caller handles gaps.
        """
        result: dict[str, SectorClassification] = {}
        if not _LSEG_AVAILABLE or ld is None:
            return result

        fields = [
            self._config.economic_sector_field,
            self._config.business_sector_field,
            self._config.industry_field,
        ]
        try:
            df = ld.get_data(universe=batch, fields=fields)
            if df is None or df.empty:
                return result

            for _, row in df.iterrows():
                ticker = str(row.get("Instrument", "")).strip()
                if not ticker:
                    continue

                econ = str(row.get(self._config.economic_sector_field, "") or "").strip()
                biz = str(row.get(self._config.business_sector_field, "") or "").strip()
                ind = str(row.get(self._config.industry_field, "") or "").strip()

                if not econ or econ.lower() in ("nan", "none", ""):
                    continue  # Not resolved — fall through to yfinance

                result[ticker] = SectorClassification(
                    ticker=ticker,
                    economic_sector=econ,
                    business_sector=biz or econ,
                    industry=ind or biz or econ,
                    source="lseg",
                )

        except Exception as exc:  # noqa: BLE001
            logger.error(f"LSEG batch fetch failed for {batch[:3]}...: {exc}")

        return result

    def _fetch_yfinance_single(self, ticker: str) -> Optional[SectorClassification]:
        """
        Fetch sector for a single ticker via yfinance .info.

        Parameters
        ----------
        ticker : str
            yfinance-compatible ticker string (e.g. "BBCA.JK", "AAPL").

        Returns
        -------
        SectorClassification or None
            None when yfinance returns no usable sector data.
        """
        try:
            info = yf.Ticker(ticker).info
            if not info:
                return None

            sector = str(info.get(self._config.yfinance_sector_field, "") or "").strip()
            if not sector or sector.lower() in ("nan", "none", ""):
                return None

            sector = normalize_sector_label(sector, source="yfinance")
            industry = str(info.get("industry", "") or "").strip() or sector

            return SectorClassification(
                ticker=ticker,
                economic_sector=sector,
                business_sector=sector,
                industry=industry,
                source="yfinance",
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning(f"yfinance sector lookup failed for {ticker}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch(self, tickers: list[str]) -> dict[str, SectorClassification]:
        """
        Fetch sector classifications for all tickers.

        Strategy:

        1. Check DataCache (TTL = config.cache_ttl_seconds).
        2. Try LSEG in batches of config.batch_size using ld.get_data().
        3. For any ticker where LSEG returned null/empty, fall back to
           yf.Ticker(t).info.get(config.yfinance_sector_field) if
           config.fallback_to_yfinance is True.
        4. Tickers still unresolved get config.unknown_sector_label.
        5. Cache the merged result.

        Parameters
        ----------
        tickers : list[str]
            List of RIC or yfinance-compatible ticker strings.

        Returns
        -------
        dict[str, SectorClassification]
            Keyed by ticker. Every input ticker guaranteed to appear.

        Raises
        ------
        LSEGConnectionError
            If LSEG is unavailable AND config.fallback_to_yfinance is False.
        """
        if not tickers:
            return {}

        cache_key = self._make_cache_key(tickers)
        cached = self._cache.get(key=cache_key, ttl=self._config.cache_ttl_seconds)
        if cached is not None:
            logger.debug(f"Sector classifications loaded from cache ({len(cached)} tickers).")
            return cached

        if not _LSEG_AVAILABLE and not self._config.fallback_to_yfinance:
            raise LSEGConnectionError(
                "LSEG Data Library is not installed and fallback_to_yfinance is False. "
                "Install with: pip install lseg-data>=2.0.0"
            )

        classifications: dict[str, SectorClassification] = {}

        # ── Step 1: LSEG batched fetch ──────────────────────────────────
        if _LSEG_AVAILABLE:
            for start in range(0, len(tickers), self._config.batch_size):
                batch = tickers[start : start + self._config.batch_size]
                batch_result = self._fetch_lseg_batch(batch)
                classifications.update(batch_result)
                logger.debug(
                    f"LSEG batch {start//self._config.batch_size + 1}: "
                    f"{len(batch_result)}/{len(batch)} resolved."
                )

        # ── Step 2: yfinance fallback for unresolved tickers ───────────
        unresolved = [t for t in tickers if t not in classifications]
        if unresolved:
            if not self._config.fallback_to_yfinance:
                logger.warning(
                    f"{len(unresolved)} tickers unresolved by LSEG and "
                    "fallback_to_yfinance is False. Marking as unknown."
                )
            else:
                logger.info(
                    f"Falling back to yfinance for {len(unresolved)} unresolved tickers."
                )
                for ticker in unresolved:
                    classification = self._fetch_yfinance_single(ticker)
                    if classification is not None:
                        classifications[ticker] = classification

        # ── Step 3: Fill remaining with unknown ─────────────────────────
        for ticker in tickers:
            if ticker not in classifications:
                classifications[ticker] = SectorClassification(
                    ticker=ticker,
                    economic_sector=self._config.unknown_sector_label,
                    business_sector=self._config.unknown_sector_label,
                    industry=self._config.unknown_sector_label,
                    source="unknown",
                )

        # ── Step 3b: Apply IDX sector overrides ─────────────────────────
        # Applied after all classification is complete.
        for ticker, override_sector in IDX_SECTOR_OVERRIDES.items():
            if ticker in classifications:
                original = classifications[ticker].economic_sector
                if original != override_sector:
                    logger.info(
                        f"IDX sector override applied: {ticker} "
                        f"{original!r} → {override_sector!r}"
                    )
                    classifications[ticker].economic_sector = override_sector
                    classifications[ticker].source = (
                        classifications[ticker].source + "+override"
                    )

        # ── Step 4: Cache and return ────────────────────────────────────
        self._cache.set(
            data=classifications,
            key=cache_key,
            ttl=self._config.cache_ttl_seconds,
            data_type="historical",
        )
        logger.info(
            f"Sector fetch complete: {sum(1 for c in classifications.values() if c.source != 'unknown')}/"
            f"{len(tickers)} tickers classified."
        )
        return classifications

    def to_sector_map(
        self,
        classifications: dict[str, SectorClassification],
        level: str = "economic",
    ) -> dict[str, str]:
        """
        Collapse full classifications to a flat {ticker: sector_name} mapping.

        Parameters
        ----------
        classifications : dict[str, SectorClassification]
            Output of fetch().
        level : str
            One of ``"economic"`` | ``"business"`` | ``"industry"``.

        Returns
        -------
        dict[str, str]
            Flat ticker → sector-name mapping at the requested hierarchy level.

        Raises
        ------
        ValueError
            If ``level`` is not one of the three accepted values.
        """
        valid_levels = {"economic", "business", "industry"}
        if level not in valid_levels:
            raise ValueError(
                f"level must be one of {sorted(valid_levels)}, got '{level}'."
            )

        mapping: dict[str, str] = {}
        for ticker, cls in classifications.items():
            if level == "economic":
                mapping[ticker] = cls.economic_sector
            elif level == "business":
                mapping[ticker] = cls.business_sector
            else:
                mapping[ticker] = cls.industry
        return mapping

    def get_unique_sectors(self, sector_map: dict[str, str]) -> list[str]:
        """
        Return sorted list of unique sector names, excluding the unknown label.

        Parameters
        ----------
        sector_map : dict[str, str]
            Output of to_sector_map().

        Returns
        -------
        list[str]
            Alphabetically sorted sector names with the unknown label omitted.
        """
        return sorted(
            {
                sector
                for sector in sector_map.values()
                if sector != self._config.unknown_sector_label
            }
        )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def _smoke_test() -> None:
        import numpy as np
        import pandas as pd

        np.random.seed(42)

        config = LSEGSectorConfig(
            fallback_to_yfinance=True,
            cache_ttl_seconds=300,
            unknown_sector_label="Unknown",
        )
        fetcher = LSEGSectorFetcher(config)

        tickers = ["AAPL", "JPM", "XOM"]

        # ── Fetch classifications ────────────────────────────────────────
        t0 = time.time()
        classifications = fetcher.fetch(tickers)
        elapsed = time.time() - t0

        assert isinstance(classifications, dict), "fetch() must return a dict"
        assert set(classifications.keys()) == set(tickers), (
            f"All input tickers must appear in output. "
            f"Got: {set(classifications.keys())}"
        )
        print(f"  fetch() returned {len(classifications)} entries in {elapsed:.1f}s")

        for ticker, cls in classifications.items():
            assert isinstance(cls, SectorClassification), (
                f"{ticker}: expected SectorClassification, got {type(cls)}"
            )
            assert cls.ticker == ticker, f"ticker mismatch: {cls.ticker} != {ticker}"
            assert cls.source in {"lseg", "yfinance", "unknown"}, (
                f"Unexpected source: {cls.source}"
            )
            print(
                f"  {ticker}: {cls.economic_sector} "
                f"({cls.business_sector}) [{cls.source}]"
            )

        # ── to_sector_map ────────────────────────────────────────────────
        for level in ("economic", "business", "industry"):
            sector_map = fetcher.to_sector_map(classifications, level=level)
            assert set(sector_map.keys()) == set(tickers), (
                f"sector_map must cover all tickers at level '{level}'"
            )
            assert all(isinstance(v, str) for v in sector_map.values()), (
                "All sector values must be strings"
            )
        print("  to_sector_map() passed for all three levels")

        # ── Invalid level raises ValueError ─────────────────────────────
        try:
            fetcher.to_sector_map(classifications, level="nonexistent")
            raise AssertionError("Expected ValueError for invalid level")
        except ValueError:
            pass
        print("  to_sector_map() raises ValueError on invalid level")

        # ── get_unique_sectors ───────────────────────────────────────────
        sector_map = fetcher.to_sector_map(classifications, level="economic")
        unique = fetcher.get_unique_sectors(sector_map)
        assert isinstance(unique, list), "get_unique_sectors() must return a list"
        assert unique == sorted(unique), "get_unique_sectors() must be sorted"
        assert config.unknown_sector_label not in unique, (
            "Unknown label must be excluded from unique sectors"
        )
        print(f"  get_unique_sectors(): {unique}")

        # ── Cache round-trip ─────────────────────────────────────────────
        classifications_cached = fetcher.fetch(tickers)
        assert classifications_cached.keys() == classifications.keys(), (
            "Cached fetch must return same keys"
        )
        print("  Cache round-trip passed")

        print("\n✓ [LSEGSectorFetcher] smoke test passed")

    _smoke_test()
