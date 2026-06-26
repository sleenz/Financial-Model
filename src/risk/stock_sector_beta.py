"""
Stock-to-sector beta with circularity correction.

Each stock's beta is estimated relative to its sector ETF (e.g., XLK for
Technology) using OLS regression. For stocks with ETF weight above
CIRCULARITY_THRESHOLD, ETF returns are adjusted to remove the stock's own
contribution (ETF-ex-stock formula), eliminating circular bias.

Indonesian tickers (.JK suffix) use ^JKSE as market proxy — no sector ETFs
available for the IDX market.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_ETF_MAP: dict[str, str] = {
    # GICS labels (yfinance fallback)
    "Technology":               "XLK",
    "Information Technology":   "XLK",
    "Financials":               "XLF",
    "Financial Services":       "XLF",
    "Consumer Discretionary":   "XLY",
    "Consumer Cyclical":        "XLY",
    "Consumer Staples":         "XLP",
    "Consumer Defensive":       "XLP",
    "Industrials":              "XLI",
    "Healthcare":               "XLV",
    "Health Care":              "XLV",
    "Real Estate":              "XLRE",
    "Basic Materials":          "XLB",
    "Materials":                "XLB",
    "Energy":                   "XLE",
    "Utilities":                "XLU",
    "Communication Services":   "XLC",
    "Telecommunications":       "XLC",
    # TRBC labels (from LSEG)
    "Consumer Cyclicals":           "XLY",
    "Consumer Non-Cyclicals":       "XLP",
    "Telecommunication Services":   "XLC",
}

CIRCULARITY_THRESHOLD = 0.10   # 10% — stocks above this get ETF-ex-stock correction

IDX_TICKER_SUFFIX = ".JK"
IDX_MARKET_PROXY = "^JKSE"

KNOWN_DOMINANT_WEIGHTS: dict[tuple[str, str], float] = {
    ("NVDA", "XLK"): 0.1307,
    ("AAPL", "XLK"): 0.1167,
    ("MSFT", "XLK"): 0.0852,
    ("AMZN", "XLY"): 0.2756,
    ("TSLA", "XLY"): 0.2004,
}


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StockBetaEntry:
    ticker: str
    sector: str
    etf_proxy: str                  # "XLK", "^JKSE", etc.
    beta: float
    r_squared: Optional[float]
    circularity_corrected: bool
    stock_weight_in_etf: Optional[float]
    source: str                     # 'sector_etf' | 'etf_ex_stock' | 'market_proxy' | 'fallback'
    n_observations: int
    warning: Optional[str]          # populated when beta is unreliable


@dataclass
class StockBetaResult:
    entries: dict[str, StockBetaEntry]
    computed_at: str
    data_start: str
    data_end: str
    n_fallbacks: int


# ─────────────────────────────────────────────────────────────────────────────
# ETF weight fetcher
# ─────────────────────────────────────────────────────────────────────────────

def get_stock_weight_in_etf(ticker: str, etf: str) -> Optional[float]:
    """
    Returns stock's weight in ETF as float (0.0–1.0), or None.

    Tries yfinance ETF holdings first, then falls back to KNOWN_DOMINANT_WEIGHTS.
    Returns None when weight cannot be determined — treat as non-dominant.
    """
    try:
        import yfinance as yf
        etf_obj = yf.Ticker(etf)
        holdings = etf_obj.funds_data.top_holdings
        if holdings is not None and not holdings.empty and ticker in holdings.index:
            weight = float(holdings.loc[ticker, "Holding Percent"])
            logger.debug(f"[WEIGHT] {ticker}/{etf}: weight={weight:.4f} source=yfinance_holdings")
            return weight
    except Exception as exc:
        logger.debug(f"[WEIGHT] {ticker}/{etf}: yfinance_holdings failed — {exc}")

    key = (ticker, etf)
    if key in KNOWN_DOMINANT_WEIGHTS:
        weight = KNOWN_DOMINANT_WEIGHTS[key]
        logger.debug(f"[WEIGHT] {ticker}/{etf}: weight={weight:.4f} source=known_dominant_weights")
        return weight

    logger.debug(f"[WEIGHT] {ticker}/{etf}: source=unknown, returning None (non-dominant)")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Circularity correction
# ─────────────────────────────────────────────────────────────────────────────

def compute_etf_ex_stock_returns(
    stock_returns: pd.Series,
    etf_returns: pd.Series,
    stock_weight_in_etf: float,
) -> pd.Series:
    """
    Removes stock's own contribution from ETF returns.

    Formula: R_etf_ex = (R_etf - w * R_stock) / (1 - w)

    Eliminates circular bias in the OLS regression — prevents inflated beta
    for dominant holdings like NVDA in XLK.
    """
    w = stock_weight_in_etf
    logger.debug(
        f"[CIRC-CORR] weight={w:.4f}, "
        f"etf_mean={etf_returns.mean():.6f}, etf_std={etf_returns.std():.6f}"
    )
    aligned = pd.concat(
        [stock_returns.rename("stock"), etf_returns.rename("etf")], axis=1
    ).dropna()
    etf_ex = (aligned["etf"] - w * aligned["stock"]) / (1.0 - w)
    corrected = etf_ex.rename(etf_returns.name)
    logger.debug(
        f"[CIRC-CORR] corrected_mean={corrected.mean():.6f}, "
        f"corrected_std={corrected.std():.6f}, "
        f"n_dropped={len(stock_returns) + len(etf_returns) - 2 * len(aligned)}"
    )
    return corrected


# ─────────────────────────────────────────────────────────────────────────────
# Core OLS beta estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_sector_relative_beta(
    ticker: str,
    sector: str,
    stock_returns: pd.Series,
    etf_prices: dict[str, pd.Series],
    min_observations: int = 120,
) -> StockBetaEntry:
    """
    Per-stock OLS beta vs its sector ETF, with optional circularity correction.

    Steps:
    1. Determine ETF proxy from SECTOR_ETF_MAP (IDX tickers → ^JKSE)
    2. Compute ETF returns from preloaded price series
    3. Check dominance → apply ETF-ex-stock correction if needed
    4. Run OLS, extract beta and R²
    5. Return StockBetaEntry with full metadata
    """
    # Step 1: Determine ETF proxy
    is_idx = ticker.upper().endswith(IDX_TICKER_SUFFIX.upper())
    if is_idx:
        etf = IDX_MARKET_PROXY
        source = "market_proxy"
        logger.debug(f"[BETA] {ticker}: IDX ticker → ETF={etf}")
    else:
        etf = SECTOR_ETF_MAP.get(sector)
        if etf is None:
            logger.warning(f"[BETA] {ticker}: sector='{sector}' not in SECTOR_ETF_MAP → fallback")
            return StockBetaEntry(
                ticker=ticker, sector=sector, etf_proxy="unknown",
                beta=1.0, r_squared=None, circularity_corrected=False,
                stock_weight_in_etf=None, source="fallback",
                n_observations=0, warning=f"Sector '{sector}' not mapped to ETF",
            )
        source = "sector_etf"

    # Step 2: Get ETF price series and compute returns
    etf_price_series = etf_prices.get(etf)
    if etf_price_series is None or etf_price_series.empty:
        logger.warning(f"[BETA] {ticker}: ETF={etf} unavailable → fallback")
        return StockBetaEntry(
            ticker=ticker, sector=sector, etf_proxy=etf,
            beta=1.0, r_squared=None, circularity_corrected=False,
            stock_weight_in_etf=None, source="fallback",
            n_observations=0, warning=f"ETF '{etf}' price series not available",
        )
    etf_returns = etf_price_series.pct_change().dropna()

    # Step 3: Circularity check and correction
    circularity_corrected = False
    stock_weight_in_etf = None
    benchmark_returns = etf_returns

    if not is_idx:
        weight = get_stock_weight_in_etf(ticker, etf)
        stock_weight_in_etf = weight
        is_dominant = weight is not None and weight > CIRCULARITY_THRESHOLD
        logger.debug(
            f"[BETA] {ticker}: ETF={etf}, dominant={is_dominant}, weight={weight}"
        )
        if is_dominant:
            try:
                benchmark_returns = compute_etf_ex_stock_returns(
                    stock_returns, etf_returns, weight
                )
                circularity_corrected = True
                source = "etf_ex_stock"
            except Exception as exc:
                logger.warning(
                    f"[BETA] {ticker}: circularity correction failed ({exc}) — using raw ETF"
                )

    # Step 4: OLS regression
    try:
        import statsmodels.api as sm

        aligned = pd.concat(
            [stock_returns.rename("stock"), benchmark_returns.rename("benchmark")],
            axis=1,
        ).dropna()
        n_obs = len(aligned)

        if n_obs == 0:
            raise ValueError("No aligned observations after dropna")

        date_start = (
            str(aligned.index[0].date())
            if hasattr(aligned.index[0], "date")
            else str(aligned.index[0])
        )
        date_end = (
            str(aligned.index[-1].date())
            if hasattr(aligned.index[-1], "date")
            else str(aligned.index[-1])
        )
        logger.debug(
            f"[BETA] {ticker}: aligned observations={n_obs}, "
            f"date_range={date_start}→{date_end}"
        )

        X = sm.add_constant(aligned["benchmark"])
        model = sm.OLS(aligned["stock"], X).fit()
        beta = float(model.params["benchmark"])
        r_squared = float(model.rsquared)
        logger.debug(
            f"[BETA] {ticker}: beta={beta:.4f}, R²={r_squared:.4f}, source={source}"
        )

        warnings: list[str] = []
        if n_obs < min_observations:
            warnings.append(f"Only {n_obs} observations (min={min_observations})")
        if r_squared < 0.10:
            warnings.append(f"Low R²={r_squared:.3f} (sector ETF explains < 10%)")
        if abs(beta) > 3.0:
            warnings.append(f"Implausible |beta|={beta:.2f}")
        if beta < 0:
            warnings.append(f"Negative beta={beta:.4f} (hedging relationship)")
        for w in warnings:
            logger.warning(f"[BETA] {ticker}: WARNING — {w}")

        return StockBetaEntry(
            ticker=ticker, sector=sector, etf_proxy=etf,
            beta=beta, r_squared=r_squared,
            circularity_corrected=circularity_corrected,
            stock_weight_in_etf=stock_weight_in_etf,
            source=source,
            n_observations=n_obs,
            warning=" | ".join(warnings) if warnings else None,
        )

    except Exception as exc:
        import traceback
        logger.error(
            f"[BETA] {ticker}: OLS failed — {exc}\n{traceback.format_exc()}"
        )
        return StockBetaEntry(
            ticker=ticker, sector=sector, etf_proxy=etf,
            beta=1.0, r_squared=None,
            circularity_corrected=False,
            stock_weight_in_etf=stock_weight_in_etf,
            source="fallback",
            n_observations=0,
            warning=f"OLS failed: {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ETF download helper
# ─────────────────────────────────────────────────────────────────────────────

def _download_etf_prices(
    etf_list: list[str],
    start_date: str,
    end_date: str,
) -> dict[str, pd.Series]:
    """
    Downloads ETF price series via yfinance. Returns dict of {etf: price_Series}.
    Any ETF that fails to download is simply omitted — caller falls back to beta=1.0.
    """
    if not etf_list:
        return {}

    import yfinance as yf

    # Add a 5-day buffer before start so pct_change() has enough data from day 1
    try:
        start_buffered = (pd.Timestamp(start_date) - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    except Exception:
        start_buffered = start_date

    try:
        raw = yf.download(
            etf_list,
            start=start_buffered,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        logger.error(f"[BETA BATCH] yf.download failed: {exc}")
        return {}

    if raw.empty:
        logger.warning(f"[BETA BATCH] yf.download returned empty DataFrame for {etf_list}")
        return {}

    result: dict[str, pd.Series] = {}
    # yfinance 0.2.x+ returns MultiIndex columns: (Price, Ticker) or (Ticker, Price)
    # Handle both single and multi-ticker cases
    for etf in etf_list:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                # Try (Price, Ticker) layout first
                if ("Close", etf) in raw.columns:
                    series = raw[("Close", etf)].dropna()
                elif (etf, "Close") in raw.columns:
                    series = raw[(etf, "Close")].dropna()
                else:
                    # Try selecting via level
                    close = raw.xs("Close", axis=1, level=0) if "Close" in raw.columns.get_level_values(0) else raw.xs("Close", axis=1, level=1)
                    series = close[etf].dropna() if etf in close.columns else pd.Series(dtype=float)
            else:
                series = raw["Close"].dropna() if len(etf_list) == 1 else raw["Close"].get(etf, pd.Series(dtype=float)).dropna()

            if not series.empty:
                result[etf] = series
                logger.debug(f"[BETA BATCH] {etf}: {len(series)} price observations")
            else:
                logger.warning(f"[BETA BATCH] {etf}: empty after download")
        except Exception as exc:
            logger.warning(f"[BETA BATCH] {etf}: extraction failed — {exc}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch computation entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_stock_betas(
    tickers: list[str],
    sector_map: dict[str, str],
    stock_returns: pd.DataFrame,
    start_date: str,
    end_date: str,
    min_observations: int = 120,
) -> StockBetaResult:
    """
    Fetches all required ETF price series once, then computes OLS beta for
    every ticker against its sector ETF (with circularity correction for
    dominant holdings).

    Falls back gracefully per-ETF if download fails, and per-ticker if OLS
    fails. A >50% fallback rate triggers an ERROR log.
    """
    # Identify unique ETFs needed
    unique_etfs: set[str] = set()
    for ticker in tickers:
        if ticker.upper().endswith(IDX_TICKER_SUFFIX.upper()):
            unique_etfs.add(IDX_MARKET_PROXY)
        else:
            sector = sector_map.get(ticker)
            etf = SECTOR_ETF_MAP.get(sector) if sector else None
            if etf:
                unique_etfs.add(etf)

    etf_list = sorted(unique_etfs)
    logger.debug(f"[BETA BATCH] ETFs to fetch: {etf_list}")

    etf_prices = _download_etf_prices(etf_list, start_date, end_date)
    logger.info(
        f"[BETA BATCH] Fetched ETF prices: {etf_list}, "
        f"available={list(etf_prices)}"
    )
    missing_etfs = [e for e in etf_list if e not in etf_prices]
    if missing_etfs:
        logger.warning(f"[BETA BATCH] Missing ETFs (download failed): {missing_etfs}")

    # Compute per-ticker betas
    entries: dict[str, StockBetaEntry] = {}
    for ticker in tickers:
        sector = sector_map.get(ticker, "Unknown")
        if ticker not in stock_returns.columns:
            logger.warning(f"[BETA] {ticker}: not in stock_returns → fallback")
            entries[ticker] = StockBetaEntry(
                ticker=ticker, sector=sector, etf_proxy="unknown",
                beta=1.0, r_squared=None, circularity_corrected=False,
                stock_weight_in_etf=None, source="fallback",
                n_observations=0, warning="Ticker not in stock_returns",
            )
            continue

        ticker_returns = stock_returns[ticker].dropna()
        if ticker_returns.empty:
            logger.warning(f"[BETA] {ticker}: no return data → fallback")
            entries[ticker] = StockBetaEntry(
                ticker=ticker, sector=sector, etf_proxy="unknown",
                beta=1.0, r_squared=None, circularity_corrected=False,
                stock_weight_in_etf=None, source="fallback",
                n_observations=0, warning="No return data",
            )
            continue

        entries[ticker] = compute_sector_relative_beta(
            ticker=ticker,
            sector=sector,
            stock_returns=ticker_returns,
            etf_prices=etf_prices,
            min_observations=min_observations,
        )

    n_fallbacks = sum(1 for e in entries.values() if e.source == "fallback")

    # Summary table
    logger.info("[BETA SUMMARY]")
    logger.info(
        f"{'ticker':<10} | {'sector':<26} | {'etf':<5} | "
        f"{'beta':>7} | {'R²':>6} | {'corrected':<10} | source"
    )
    logger.info("-" * 83)
    for ticker, entry in entries.items():
        logger.info(
            f"{entry.ticker:<10} | {entry.sector:<26} | {entry.etf_proxy:<5} | "
            f"{entry.beta:>7.4f} | {(entry.r_squared or 0.0):>6.3f} | "
            f"{str(entry.circularity_corrected):<10} | {entry.source}"
        )
    logger.info(f"[BETA SUMMARY] Fallbacks: {n_fallbacks}/{len(tickers)}")
    if n_fallbacks > len(tickers) * 0.5:
        logger.error(
            f"[BETA SUMMARY] >50% fallback rate ({n_fallbacks}/{len(tickers)}) "
            "— ETF fetch likely failed. All betas default to 1.0."
        )

    return StockBetaResult(
        entries=entries,
        computed_at=pd.Timestamp.now().isoformat(),
        data_start=start_date,
        data_end=end_date,
        n_fallbacks=n_fallbacks,
    )
