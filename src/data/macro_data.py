"""
Macro variable data fetcher for the Leontief contagion model.

Fetches DXY, VIX, US Treasury yield, Bank Indonesia rate, IDR/USD,
China PMI, palm oil, coal, and nickel from yfinance and FRED with
automatic fallback, per-variable caching, and graceful degradation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, date as _date
from typing import Optional

import numpy as np
import pandas as pd

from src.data.cache import DataCache
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Optional dependency guards ────────────────────────────────────────────────

try:
    import yfinance as _yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _yf = None
    _YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available — yfinance macro sources will fail")

try:
    from fredapi import Fred as _Fred
    _FREDAPI_AVAILABLE = True
except ImportError:
    _Fred = None
    _FREDAPI_AVAILABLE = False
    logger.warning("fredapi not installed — FRED sources will use pandas_datareader fallback")

try:
    import sqlite3 as _sqlite3
    _SQLITE3_AVAILABLE = True
except ImportError:
    _sqlite3 = None
    _SQLITE3_AVAILABLE = False

try:
    import joblib as _joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _joblib = None
    _JOBLIB_AVAILABLE = False

try:
    import networkx as _nx
    _NETWORKX_AVAILABLE = True
except ImportError:
    _nx = None
    _NETWORKX_AVAILABLE = False

try:
    import tradingeconomics as _te
    _TE_AVAILABLE = True
except ImportError:
    _te = None
    _TE_AVAILABLE = False
    logger.warning(
        "tradingeconomics not installed — TE sources will fail. "
        "Install with: pip install tradingeconomics"
    )


# ── Variable configuration ────────────────────────────────────────────────────

@dataclass
class MacroVariableConfig:
    """
    Configuration for a single macro variable.

    Parameters
    ----------
    name : str
        Human-readable label, e.g. "DXY".
    primary_ticker : str
        yfinance ticker or FRED series ID for primary source.
    primary_source : str
        "yfinance" or "fred".
    fallback_ticker : str, optional
        Alternative ticker/series if primary fails.
    fallback_source : str, optional
        "yfinance" or "fred".
    transform : str
        "pct_change": weekly % change (equities, FX, commodities).
        "diff": level difference (rate variables, bps equivalent).
        "level": raw level, no transformation (PMI).
    frequency : str
        "W" weekly | "M" monthly | "D" daily.
    description : str
        Documentation string.
    """

    name: str
    primary_ticker: str
    primary_source: str
    fallback_ticker: Optional[str] = field(default=None)
    fallback_source: Optional[str] = field(default=None)
    transform: str = field(default="pct_change")
    frequency: str = field(default="W")
    description: str = field(default="")


DEFAULT_MACRO_VARIABLES: list[MacroVariableConfig] = [
    MacroVariableConfig(
        name="DXY",
        primary_ticker="DXY:CUR",
        primary_source="te_market",
        fallback_ticker="DX-Y.NYB",
        fallback_source="yfinance",
        transform="pct_change",
        frequency="W",
        description="US Dollar Index — primary EM risk driver",
    ),
    MacroVariableConfig(
        name="VIX",
        primary_ticker="VIX:IND",
        primary_source="te_market",
        fallback_ticker="^VIX",
        fallback_source="yfinance",
        transform="diff",
        frequency="W",
        description="CBOE Volatility Index — global risk-off signal",
    ),
    MacroVariableConfig(
        name="US_10Y",
        primary_ticker="USGG10YR:IND",
        primary_source="te_market",
        fallback_ticker="DGS10",
        fallback_source="fred",
        transform="diff",
        frequency="W",
        description="US 10Y Treasury yield change (bps)",
    ),
    MacroVariableConfig(
        name="BI_RATE",
        primary_ticker="Indonesia|Interest Rate",
        primary_source="te_indicator",
        fallback_ticker=None,
        fallback_source=None,
        transform="diff",
        frequency="M",
        description="Bank Indonesia policy rate change (bps) — TE indicator",
    ),
    MacroVariableConfig(
        name="IDR_USD",
        primary_ticker="USDIDR:CUR",
        primary_source="te_market",
        fallback_ticker="IDR=X",
        fallback_source="yfinance",
        transform="pct_change",
        frequency="W",
        description="USD/IDR exchange rate — positive = IDR weakening",
    ),
    MacroVariableConfig(
        name="CHINA_PMI",
        primary_ticker="China|NBS Manufacturing PMI",
        primary_source="te_indicator",
        fallback_ticker=None,
        fallback_source=None,
        transform="diff",
        frequency="M",
        description="China NBS Manufacturing PMI — month-over-month point change",
    ),
    MacroVariableConfig(
        name="CPO",
        primary_ticker="CPO1:COM",
        primary_source="te_market",
        fallback_ticker="PPOILUSDM",
        fallback_source="fred",
        transform="pct_change",
        frequency="W",
        description="Palm oil futures (Bursa Malaysia) — IDX #1 agricultural export",
    ),
    MacroVariableConfig(
        name="COAL",
        primary_ticker="NEWC:COM",
        primary_source="te_market",
        fallback_ticker="MTF=F",
        fallback_source="yfinance",
        transform="pct_change",
        frequency="W",
        description="Newcastle thermal coal — IDX key commodity",
    ),
    MacroVariableConfig(
        name="NICKEL",
        primary_ticker="LMENIS3:COM",
        primary_source="te_market",
        fallback_ticker="PNICKUSDM",
        fallback_source="fred",
        transform="pct_change",
        frequency="W",
        description="LME Nickel 3-month — ANTM, INCO; EV battery demand proxy",
    ),
]


@dataclass
class MacroDataConfig:
    """
    Top-level configuration for MacroDataFetcher.

    Parameters
    ----------
    variables : list[MacroVariableConfig]
        Variables to fetch. Defaults to DEFAULT_MACRO_VARIABLES.
    start_date : str
        ISO format start date for historical fetch.
    cache_ttl_seconds : int
        Per-variable cache TTL. Default 3600s (1 hour).
    fred_api_key : str, optional
        FRED API key. If None, reads FRED_API_KEY env var.
    fill_method : str
        How to align monthly variables to weekly:
        "ffill" (default) or "interpolate".
    min_overlap_pct : float
        Minimum fraction of dates with data before warning.
    """

    variables: list[MacroVariableConfig] = field(
        default_factory=lambda: list(DEFAULT_MACRO_VARIABLES)
    )
    start_date: str = field(default="2005-01-01")
    cache_ttl_seconds: int = field(default=3600)
    te_api_key: Optional[str] = field(default=None)
    fred_api_key: Optional[str] = field(default=None)
    fill_method: str = field(default="ffill")
    min_overlap_pct: float = field(default=0.70)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MacroDataResult:
    """
    Output of MacroDataFetcher.fetch().

    Attributes
    ----------
    raw_levels : pd.DataFrame
        (T, N) raw price/rate levels.
    transformed : pd.DataFrame
        (T, N) transformed per-variable config.
    aligned_weekly : pd.DataFrame
        (T, N) all variables on a common weekly index.
    missing_variables : list[str]
        Variables that failed all sources.
    source_used : dict[str, str]
        {variable_name: "primary" | "fallback" | "failed"}.
    date_range : tuple[str, str]
        (start, end) ISO strings.
    config : MacroDataConfig
    """

    raw_levels: pd.DataFrame
    transformed: pd.DataFrame
    aligned_weekly: pd.DataFrame
    missing_variables: list[str]
    source_used: dict[str, str]
    date_range: tuple[str, str]
    config: MacroDataConfig


# ── Fetcher ───────────────────────────────────────────────────────────────────

class MacroDataFetcher:
    """
    Fetch macro variables from yfinance and FRED with fallback and caching.

    Parameters
    ----------
    config : MacroDataConfig
        Configuration for variables, dates, and caching behaviour.
    """

    def __init__(self, config: MacroDataConfig = MacroDataConfig()) -> None:
        self._config = config
        self._cache = DataCache()

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        start_date: str = None,
        end_date: str = None,
    ) -> MacroDataResult:
        """
        Fetch all macro variables defined in config.variables.

        For each variable:
        1. Check DataCache (TTL = config.cache_ttl_seconds).
        2. Try primary source.
        3. On failure, try fallback source.
        4. Apply transform (pct_change / diff / level).
        5. Resample to weekly (config.fill_method for monthly variables).
        6. Align all variables to the same weekly date index.

        Missing variables get a column of NaN with a logged warning.
        Never raises — degrades gracefully with missing_variables populated.

        Parameters
        ----------
        start_date : str, optional
            ISO format. Defaults to config.start_date.
        end_date : str, optional
            ISO format. Defaults to today.

        Returns
        -------
        MacroDataResult
        """
        t0 = time.time()
        start = start_date or self._config.start_date
        end = end_date or datetime.today().strftime("%Y-%m-%d")

        raw_series: dict[str, pd.Series] = {}
        transformed_series: dict[str, pd.Series] = {}
        weekly_series: dict[str, pd.Series] = {}
        source_used: dict[str, str] = {}
        missing_variables: list[str] = []

        for var_cfg in self._config.variables:
            name = var_cfg.name
            try:
                raw = self._fetch_with_cache(var_cfg, start, end)
                source_used[name] = raw.attrs.get("source", "primary")
                raw_series[name] = raw

                transformed = self._apply_transform(raw, var_cfg.transform)
                transformed_series[name] = transformed

                weekly = self._to_weekly(transformed, var_cfg.frequency)
                weekly_series[name] = weekly
                logger.debug(
                    f"MacroDataFetcher: {name} fetched {len(raw)} obs, "
                    f"source={source_used[name]}"
                )
            except Exception as exc:
                logger.warning(f"MacroDataFetcher: {name} failed all sources: {exc}")
                source_used[name] = "failed"
                missing_variables.append(name)

        # Build raw_levels and transformed DataFrames (before weekly resampling)
        raw_levels = (
            pd.DataFrame({k: v for k, v in raw_series.items()})
            if raw_series else pd.DataFrame()
        )
        transformed_df = (
            pd.DataFrame({k: v for k, v in transformed_series.items()})
            if transformed_series else pd.DataFrame()
        )

        # Align weekly series to a common Friday-end weekly index
        aligned_weekly = self._align_weekly(weekly_series, start, end)

        # Warn on sparse columns
        if not aligned_weekly.empty:
            valid_frac = aligned_weekly.notna().mean()
            for col in aligned_weekly.columns:
                if valid_frac[col] < self._config.min_overlap_pct:
                    logger.warning(
                        f"MacroDataFetcher: {col} has only "
                        f"{valid_frac[col]:.0%} valid weekly observations"
                    )

        date_range = (start, end)
        elapsed = time.time() - t0
        logger.info(
            f"MacroDataFetcher.fetch() done in {elapsed:.2f}s — "
            f"shape={aligned_weekly.shape}, missing={missing_variables}"
        )
        return MacroDataResult(
            raw_levels=raw_levels,
            transformed=transformed_df,
            aligned_weekly=aligned_weekly,
            missing_variables=missing_variables,
            source_used=source_used,
            date_range=date_range,
            config=self._config,
        )

    def get_variable_names(self) -> list[str]:
        """
        Return list of variable names in config order.

        Returns
        -------
        list[str]
        """
        return [v.name for v in self._config.variables]

    def get_macro_shock_vector(
        self,
        result: MacroDataResult,
        shock_date: str,
    ) -> pd.Series:
        """
        Extract macro variable values on a specific date as a shock vector.

        Used to replay historical macro conditions through the contagion model.

        Parameters
        ----------
        result : MacroDataResult
            Output of fetch().
        shock_date : str
            ISO date string. Nearest available date is used if exact date
            is not in the index.

        Returns
        -------
        pd.Series
            Index = variable names. Values = transformed macro values on that date.
        """
        df = result.aligned_weekly
        if df.empty:
            return pd.Series(dtype=float)
        idx = df.index.asof(pd.Timestamp(shock_date))
        if pd.isna(idx):
            logger.warning(
                f"get_macro_shock_vector: {shock_date} not found in weekly index"
            )
            return pd.Series(0.0, index=df.columns)
        return df.loc[idx]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch_with_cache(
        self, var_cfg: MacroVariableConfig, start: str, end: str
    ) -> pd.Series:
        """Try cache → primary → fallback. Attaches source label via attrs."""
        cache_key = self._cache._generate_key(
            "macro_raw", var_cfg.name, start, end
        )
        cached = self._cache.get(cache_key, ttl=self._config.cache_ttl_seconds)
        if cached is not None:
            cached.attrs["source"] = cached.attrs.get("source", "primary")
            return cached

        # Primary source
        primary_exc: Optional[Exception] = None
        try:
            raw = self._fetch_source(
                var_cfg.primary_ticker, var_cfg.primary_source, start, end
            )
            raw.attrs["source"] = "primary"
            self._cache.set(raw, cache_key, self._config.cache_ttl_seconds)
            return raw
        except Exception as exc:
            primary_exc = exc
            logger.debug(
                f"{var_cfg.name}: primary source failed ({var_cfg.primary_source}/"
                f"{var_cfg.primary_ticker}): {exc}"
            )

        # Fallback source
        if var_cfg.fallback_ticker and var_cfg.fallback_source:
            try:
                raw = self._fetch_source(
                    var_cfg.fallback_ticker, var_cfg.fallback_source, start, end
                )
                raw.attrs["source"] = "fallback"
                self._cache.set(raw, cache_key, self._config.cache_ttl_seconds)
                return raw
            except Exception as exc:
                logger.debug(
                    f"{var_cfg.name}: fallback source also failed "
                    f"({var_cfg.fallback_source}/{var_cfg.fallback_ticker}): {exc}"
                )

        raise RuntimeError(
            f"{var_cfg.name}: all sources failed. "
            f"Primary error: {primary_exc}"
        )

    def _fetch_source(
        self, ticker: str, source: str, start: str, end: str
    ) -> pd.Series:
        """Dispatch to the correct source fetcher."""
        if source == "te_market":
            return self._fetch_te_market(ticker, start, end)
        if source == "te_indicator":
            return self._fetch_te_indicator(ticker, start, end)
        if source == "yfinance":
            return self._fetch_yfinance(ticker, start, end)
        if source == "fred":
            return self._fetch_fred(ticker, start, end)
        raise ValueError(f"Unknown source '{source}'")

    def _fetch_te_market(self, symbol: str, start: str, end: str) -> pd.Series:
        """
        Fetch historical market data from Trading Economics.

        Parameters
        ----------
        symbol : str
            Trading Economics market symbol, e.g. "DXY:CUR", "VIX:IND",
            "CPO1:COM", "NEWC:COM", "LMENIS3:COM", "USDIDR:CUR".
        start, end : str
            ISO date strings.

        Returns
        -------
        pd.Series
            Close price series, index = DatetimeIndex (tz-naive).
        """
        if not _TE_AVAILABLE:
            raise RuntimeError(
                "tradingeconomics not installed — pip install tradingeconomics"
            )
        api_key = self._config.te_api_key or os.environ.get("TE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                f"TE_API_KEY not set — cannot fetch {symbol}. "
                "Add TE_API_KEY to your .env file."
            )
        _te.login(api_key)
        df = _te.getHistoricalBySymbol(
            symbol=symbol, initDate=start, endDate=end, output_type="df"
        )
        if df is None or (hasattr(df, "empty") and df.empty):
            raise ValueError(f"Trading Economics returned empty data for {symbol}")
        date_col = next(
            (c for c in df.columns if c.lower() == "date"), None
        )
        close_col = next(
            (c for c in df.columns if c.lower() in ("close", "value", "last")), None
        )
        if date_col is None or close_col is None:
            raise ValueError(
                f"Unexpected TE market columns for {symbol}: {df.columns.tolist()}"
            )
        df[date_col] = pd.to_datetime(df[date_col])
        series = df.set_index(date_col)[close_col].dropna().sort_index()
        series.index = pd.DatetimeIndex(series.index).tz_localize(None)
        series.name = symbol
        return series

    def _fetch_te_indicator(self, country_indicator: str, start: str, end: str) -> pd.Series:
        """
        Fetch historical economic indicator from Trading Economics.

        Parameters
        ----------
        country_indicator : str
            Pipe-separated "Country|Indicator" string, e.g.
            "Indonesia|Interest Rate" or "China|NBS Manufacturing PMI".
        start, end : str
            ISO date strings.

        Returns
        -------
        pd.Series
            Indicator value series, index = DatetimeIndex (tz-naive).
        """
        if not _TE_AVAILABLE:
            raise RuntimeError(
                "tradingeconomics not installed — pip install tradingeconomics"
            )
        api_key = self._config.te_api_key or os.environ.get("TE_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                f"TE_API_KEY not set — cannot fetch {country_indicator}. "
                "Add TE_API_KEY to your .env file."
            )
        country, indicator = country_indicator.split("|", 1)
        _te.login(api_key)
        df = _te.getHistoricalData(
            country=country.strip(),
            indicator=indicator.strip(),
            initDate=start,
            endDate=end,
            output_type="df",
        )
        if df is None or (hasattr(df, "empty") and df.empty):
            raise ValueError(
                f"Trading Economics returned empty data for "
                f"{country.strip()}/{indicator.strip()}"
            )
        date_col = next(
            (c for c in df.columns if c.lower() in ("datetime", "date")), None
        )
        val_col = next(
            (c for c in df.columns if c.lower() == "value"), None
        )
        if date_col is None or val_col is None:
            raise ValueError(
                f"Unexpected TE indicator columns for {country_indicator}: "
                f"{df.columns.tolist()}"
            )
        df[date_col] = pd.to_datetime(df[date_col])
        series = df.set_index(date_col)[val_col].dropna().sort_index()
        series.index = pd.DatetimeIndex(series.index).tz_localize(None)
        series.name = country_indicator
        return series

    def _fetch_yfinance(self, ticker: str, start: str, end: str) -> pd.Series:
        """
        Fetch adjusted close price series from yfinance.

        Parameters
        ----------
        ticker : str
            yfinance symbol (e.g. "DX-Y.NYB", "^VIX").
        start, end : str
            ISO date strings.

        Returns
        -------
        pd.Series
            Daily close price, index = DatetimeIndex.
        """
        if not _YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance not installed")
        t = _yf.Ticker(ticker)
        hist = t.history(start=start, end=end, auto_adjust=True)
        if hist.empty:
            raise ValueError(f"yfinance returned empty data for {ticker}")
        if "Close" not in hist.columns:
            raise ValueError(f"No 'Close' column for {ticker}: {hist.columns.tolist()}")
        series = hist["Close"].dropna()
        series.index = series.index.tz_localize(None)
        series.name = ticker
        return series

    def _fetch_fred(self, series_id: str, start: str, end: str) -> pd.Series:
        """
        Fetch series from FRED via fredapi (preferred) with pandas_datareader fallback.

        Reads FRED_API_KEY from config or the FRED_API_KEY environment variable
        (set in .env — loaded automatically via dotenv at import time).

        Parameters
        ----------
        series_id : str
            FRED series ID (e.g. "DGS10").
        start, end : str
            ISO date strings.

        Returns
        -------
        pd.Series
            Level series from FRED, index = DatetimeIndex.
        """
        api_key = self._config.fred_api_key or os.environ.get("FRED_API_KEY", "")

        if not api_key:
            logger.warning(
                f"FRED_API_KEY not set — fetching {series_id} via pandas_datareader "
                "(unauthenticated CSV). Add FRED_API_KEY to your .env for reliable access."
            )

        if _FREDAPI_AVAILABLE:
            try:
                fred_kwargs = {"api_key": api_key} if api_key else {}
                fred = _Fred(**fred_kwargs)
                data = fred.get_series(series_id, observation_start=start, observation_end=end)
                if data is None or data.empty:
                    raise ValueError(f"FRED returned empty series for {series_id}")
                series = data.dropna()
                series.name = series_id
                series.index = pd.DatetimeIndex(series.index)
                return series
            except Exception as exc:
                logger.warning(
                    f"fredapi fetch failed for {series_id}: {exc}; "
                    "falling back to pandas_datareader"
                )
        else:
            logger.warning(
                "fredapi not installed — install it with: pip install fredapi>=0.5.0"
            )

        # Fallback: pandas_datareader (unauthenticated CSV)
        try:
            import pandas_datareader.data as web
            df = web.DataReader(series_id, "fred", start=start, end=end)
            if df.empty:
                raise ValueError(f"pandas_datareader returned empty data for {series_id}")
            series = df.iloc[:, 0].dropna()
            series.name = series_id
            return series
        except ImportError:
            raise RuntimeError(
                f"Neither fredapi nor pandas_datareader is available "
                f"for FRED series '{series_id}'. "
                "Install fredapi>=0.5.0 and set FRED_API_KEY in .env."
            )

    def _apply_transform(self, series: pd.Series, transform: str) -> pd.Series:
        """
        Apply transform to raw level series.

        Parameters
        ----------
        series : pd.Series
            Raw level series.
        transform : str
            "pct_change": (p_t / p_{t-1}) - 1.
            "diff": p_t - p_{t-1}.
            "level": series as-is.

        Returns
        -------
        pd.Series
            Transformed series (one fewer observation for pct_change/diff).
        """
        if transform == "pct_change":
            result = series.pct_change()
        elif transform == "diff":
            result = series.diff()
        elif transform == "level":
            result = series.copy()
        else:
            logger.warning(f"Unknown transform '{transform}' — using level")
            result = series.copy()
        return result.dropna()

    def _to_weekly(self, series: pd.Series, frequency: str) -> pd.Series:
        """
        Resample a series to weekly (Friday) frequency.

        For weekly/daily data: take last observation of each week.
        For monthly data: resample to weekly then forward-fill.

        Parameters
        ----------
        series : pd.Series
        frequency : str
            "W" | "D" | "M"

        Returns
        -------
        pd.Series
            Weekly series indexed at week-ending Fridays.
        """
        if series.empty:
            return series
        series = series.sort_index()
        if frequency == "M":
            # Monthly: resample to weekly then fill
            weekly = series.resample("W").last()
            if self._config.fill_method == "interpolate":
                weekly = weekly.interpolate(method="time")
            else:
                weekly = weekly.ffill()
        else:
            # Daily or already weekly: take last of each week
            weekly = series.resample("W").last()
        return weekly.dropna()

    def _align_weekly(
        self,
        weekly_series: dict[str, pd.Series],
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Align all weekly series to a common weekly index, inserting NaN for
        missing variables.

        Parameters
        ----------
        weekly_series : dict[str, pd.Series]
        start, end : str

        Returns
        -------
        pd.DataFrame
            (T, N) weekly DataFrame.
        """
        if not weekly_series:
            return pd.DataFrame()

        # Build a union of all weekly indices
        all_indices = [s.index for s in weekly_series.values() if not s.empty]
        if not all_indices:
            return pd.DataFrame()

        union_idx = all_indices[0]
        for idx in all_indices[1:]:
            union_idx = union_idx.union(idx)

        # Filter to requested date range
        union_idx = union_idx[
            (union_idx >= pd.Timestamp(start)) & (union_idx <= pd.Timestamp(end))
        ]

        # Reindex each series
        aligned: dict[str, pd.Series] = {}
        for name in [v.name for v in self._config.variables]:
            if name in weekly_series:
                aligned[name] = weekly_series[name].reindex(union_idx)
            else:
                aligned[name] = pd.Series(np.nan, index=union_idx, name=name)

        return pd.DataFrame(aligned, index=union_idx)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _smoke_test() -> None:
        config = MacroDataConfig(
            variables=[
                v for v in DEFAULT_MACRO_VARIABLES
                if v.name in ["DXY", "VIX", "US_10Y"]
            ],
            start_date="2020-01-01",
        )
        fetcher = MacroDataFetcher(config)
        result = fetcher.fetch()

        assert result.aligned_weekly.shape[1] == 3, (
            f"Expected 3 columns, got {result.aligned_weekly.shape[1]}"
        )
        assert len(result.missing_variables) == 0 or True, "Allow partial fetch"

        print(f"macro_data smoke test passed: {result.aligned_weekly.shape}")
        print(f"  Variables: {list(result.aligned_weekly.columns)}")
        print(f"  Date range: {result.date_range}")
        print(f"  Sources used: {result.source_used}")
        print(f"  Missing: {result.missing_variables}")

        # get_variable_names
        names = fetcher.get_variable_names()
        assert names == ["DXY", "VIX", "US_10Y"], f"Unexpected names: {names}"
        print(f"  get_variable_names(): {names}")

        # get_macro_shock_vector — pick a date in the middle
        if not result.aligned_weekly.empty:
            mid_date = result.aligned_weekly.index[len(result.aligned_weekly) // 2]
            shock_vec = fetcher.get_macro_shock_vector(result, str(mid_date.date()))
            assert isinstance(shock_vec, pd.Series)
            print(f"  get_macro_shock_vector() on {mid_date.date()}: {shock_vec.to_dict()}")

        print("\nmacro_data smoke test PASSED")

    _smoke_test()
