"""
stock_valuer.py
---------------
Institutional-grade 3-stage stock valuation pipeline.

Stages
------
1. Greenblatt Magic Formula Screen  — ranks on EBIT/EV and ROIC, filters junk
2. Multi-Factor Composite Score     — graduated 0-100 score across 5 factor groups
3. Reverse DCF Validation           — implied vs. historical FCF growth check

Each stage can be run independently.  The orchestrator ``analyze_stocks``
runs the full pipeline and returns a merged DataFrame sorted by score.

Usage
-----
    from stock_valuer import analyze_stocks, print_report
    df, warnings = analyze_stocks(["AAPL", "MSFT", "GOOGL"])
    print_report(df)

Or run directly:
    python stock_valuer.py
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import brentq

try:
    from tabulate import tabulate as _tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    from rich.console import Console
    from rich.table import Table
    HAS_RICH = True
    _console = Console()
except ImportError:
    HAS_RICH = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDED_SECTORS: set[str] = {"Financials", "Utilities"}
MIN_MARKET_CAP_USD: float = 500_000_000  # $500 M

SECTOR_THRESHOLDS: dict[str, dict] = {
    "Technology": {
        "gross_margin_min": 40,
        "ev_ebit_fair": 20,
        "fcf_yield_min": 4,
    },
    "Consumer Discretionary": {
        "gross_margin_min": 30,
        "ev_ebit_fair": 14,
        "fcf_yield_min": 5,
    },
    "Healthcare": {
        "gross_margin_min": 50,
        "ev_ebit_fair": 18,
        "fcf_yield_min": 3,
    },
    "Industrials": {
        "gross_margin_min": 25,
        "ev_ebit_fair": 12,
        "fcf_yield_min": 6,
    },
    "Consumer Staples": {
        "gross_margin_min": 20,
        "ev_ebit_fair": 14,
        "fcf_yield_min": 5,
    },
}

_DEFAULT_THRESHOLDS: dict = {
    "gross_margin_min": 30,
    "ev_ebit_fair": 15,
    "fcf_yield_min": 5,
}

# Reference EV/EBITDA sector medians used for relative-value scoring.
SECTOR_EV_EBITDA_MEDIANS: dict[str, float] = {
    "Technology": 22.0,
    "Healthcare": 18.0,
    "Consumer Discretionary": 14.0,
    "Consumer Staples": 14.0,
    "Industrials": 13.0,
    "Energy": 7.0,
    "Materials": 10.0,
    "Real Estate": 20.0,
    "Communication Services": 15.0,
    "default": 14.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(value) -> Optional[float]:
    """Return float or None; filters out NaN/Inf/None."""
    if value is None:
        return None
    try:
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _coalesce(*values) -> Optional[float]:
    """
    Return the first non-None value from the argument list.

    Unlike Python's ``or`` operator, this preserves legitimate zero values:
    ``_coalesce(0.0, 1.0)`` returns ``0.0``, not ``1.0``.
    Use this for financial metrics (debt, cash, FCF) where zero is a real value.
    """
    for v in values:
        if v is not None:
            return v
    return None


def _info_val(info: dict, *keys: str) -> Optional[float]:
    """Try multiple keys in the yfinance info dict; return the first valid float."""
    for key in keys:
        val = _safe_float(info.get(key))
        if val is not None:
            return val
    return None


def _get_stmt_row(df: Optional[pd.DataFrame], *names: str) -> Optional[pd.Series]:
    """
    Return the first matching row from a financial statement DataFrame.

    The DataFrame index holds metric names; columns are date-ordered (most
    recent first in yfinance).  Returns the row sorted most-recent-first,
    or None if no name matches.
    """
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            row = df.loc[name].dropna()
            if not row.empty:
                return row.sort_index(ascending=False)
    return None


def _latest(series: Optional[pd.Series]) -> Optional[float]:
    """Most recent non-NaN value from a statement series."""
    if series is None or series.empty:
        return None
    return _safe_float(series.iloc[0])


def _ttm_sum(
    q_df: Optional[pd.DataFrame],
    a_df: Optional[pd.DataFrame],
    *names: str,
) -> Optional[float]:
    """
    Compute trailing-twelve-months value for a flow metric.

    Prefers summing the last four quarterly values; falls back to the
    most recent annual figure.
    """
    row_q = _get_stmt_row(q_df, *names)
    if row_q is not None and len(row_q) >= 4:
        return float(row_q.iloc[:4].sum())
    row_a = _get_stmt_row(a_df, *names)
    return _latest(row_a)


def _cagr(series: Optional[pd.Series], years: int) -> Optional[float]:
    """
    Compute CAGR from a series sorted most-recent-first.

    Returns None when there are fewer than ``years + 1`` data points or
    when either endpoint is non-positive.
    """
    if series is None or len(series) < years + 1:
        return None
    v_now = _safe_float(series.iloc[0])
    v_then = _safe_float(series.iloc[years])
    if v_now is None or v_then is None or v_then <= 0 or v_now <= 0:
        return None
    return (v_now / v_then) ** (1.0 / years) - 1.0


def _fetch_all(ticker: str) -> dict:
    """
    Fetch all yfinance data objects for a ticker in a single Ticker call.

    Returns a dict with keys: t, info, fin, q_fin, bs, cf, q_cf.
    The ``t`` key holds the raw Ticker object so callers can make additional
    requests (e.g. ``.history()``) without creating a second network object.
    Quarterly balance-sheet data is intentionally excluded — the annual
    balance sheet (``bs``) is used for all point-in-time balance items.
    """
    t = yf.Ticker(ticker)
    return {
        "t": t,
        "info": t.info or {},
        "fin": t.financials,
        "q_fin": t.quarterly_financials,
        "bs": t.balance_sheet,
        "cf": t.cashflow,
        "q_cf": t.quarterly_cashflow,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — GREENBLATT MAGIC FORMULA SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def magic_formula_screen(tickers: list[str]) -> pd.DataFrame:
    """
    Greenblatt Magic Formula screen: rank stocks on EBIT/EV and ROIC.

    Each stock is ranked independently on earnings yield (EBIT/EV, rank 1 =
    highest) and ROIC (rank 1 = highest).  The combined rank is the sum of
    both individual ranks; a lower combined rank is better.

    Pre-ranking filters remove:
    - Market cap < $500 M
    - Financials or Utilities sector
    - Negative EBIT
    - Total Debt > 3x EBITDA (distressed)

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to screen.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, ebit_ev, roic, ebit_ev_rank, roic_rank,
        combined_rank.  Sorted by combined_rank ascending (best first).
    """
    records: list[dict] = []
    warnings: list[str] = []

    for ticker in tickers:
        try:
            d = _fetch_all(ticker)
            info = d["info"]
            fin, q_fin = d["fin"], d["q_fin"]
            bs = d["bs"]

            # ── Filter: market cap ────────────────────────────────────────────
            mkt_cap = _info_val(info, "marketCap")
            if mkt_cap is None or mkt_cap < MIN_MARKET_CAP_USD:
                warnings.append(f"{ticker}: dropped – market cap below $500 M")
                continue

            # ── Filter: sector ────────────────────────────────────────────────
            sector = info.get("sector", "Unknown") or "Unknown"
            if sector in EXCLUDED_SECTORS:
                warnings.append(f"{ticker}: dropped – excluded sector ({sector})")
                continue

            # ── EBIT (TTM) ────────────────────────────────────────────────────
            ebit = _ttm_sum(q_fin, fin, "Operating Income", "Ebit", "EBIT")
            if ebit is None or ebit <= 0:
                warnings.append(f"{ticker}: dropped – negative or missing EBIT")
                continue

            # ── EBITDA for distress filter ────────────────────────────────────
            ebitda = _coalesce(
                _info_val(info, "ebitda"),
                _ttm_sum(q_fin, fin, "EBITDA", "Ebitda", "Normalized EBITDA"),
            )

            # ── Balance sheet items ───────────────────────────────────────────
            total_debt = _coalesce(
                _info_val(info, "totalDebt"),
                _latest(_get_stmt_row(bs, "Total Debt",
                                      "Long Term Debt And Capital Lease Obligation")),
                0.0,
            )
            cash = _coalesce(
                _info_val(info, "totalCash"),
                _latest(_get_stmt_row(
                    bs,
                    "Cash And Cash Equivalents",
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Short Term Investments",
                )),
                0.0,
            )

            # ── Filter: distressed debt ───────────────────────────────────────
            if ebitda and ebitda > 0 and total_debt > 3 * ebitda:
                warnings.append(f"{ticker}: dropped – total debt > 3x EBITDA")
                continue

            # ── EBIT / EV ─────────────────────────────────────────────────────
            ev = mkt_cap + total_debt - cash
            if ev <= 0:
                warnings.append(f"{ticker}: dropped – negative enterprise value")
                continue
            ebit_ev = ebit / ev

            # ── ROIC ──────────────────────────────────────────────────────────
            ca = _latest(_get_stmt_row(bs, "Total Current Assets", "Current Assets"))
            cl = _latest(_get_stmt_row(bs, "Total Current Liabilities",
                                        "Current Liabilities"))
            ppe = _latest(_get_stmt_row(
                bs, "Net PPE",
                "Net Property Plant And Equipment",
                "Net Property Plant Equipment",
            )) or 0.0
            st_debt = _latest(_get_stmt_row(
                bs, "Current Debt",
                "Current Debt And Capital Lease Obligation",
                "Short Term Debt",
            )) or 0.0

            if ca is None or cl is None:
                warnings.append(f"{ticker}: dropped – missing balance sheet data for ROIC")
                continue

            # Exclude cash from CA and short-term debt from CL per Greenblatt
            nwc = (ca - cash) - (cl - st_debt)
            invested_capital = nwc + ppe
            if invested_capital <= 0:
                warnings.append(f"{ticker}: dropped – non-positive invested capital")
                continue

            roic = ebit / invested_capital

            records.append({
                "ticker": ticker,
                "ebit_ev": ebit_ev,
                "roic": roic,
                "sector": sector,
            })

        except Exception as exc:
            warnings.append(f"{ticker}: error during screen – {exc}")
            logger.warning("Screen error for %s: %s", ticker, exc)

    for w in warnings:
        logger.info(w)

    if not records:
        return pd.DataFrame(
            columns=["ticker", "ebit_ev", "roic",
                     "ebit_ev_rank", "roic_rank", "combined_rank"]
        )

    df = pd.DataFrame(records)
    df["ebit_ev_rank"] = df["ebit_ev"].rank(ascending=False, method="min").astype(int)
    df["roic_rank"] = df["roic"].rank(ascending=False, method="min").astype(int)
    df["combined_rank"] = df["ebit_ev_rank"] + df["roic_rank"]
    df = df.sort_values("combined_rank").reset_index(drop=True)

    return df[["ticker", "ebit_ev", "roic",
               "ebit_ev_rank", "roic_rank", "combined_rank", "sector"]]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — MULTI-FACTOR COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def multi_factor_score(
    ticker: str,
    sector_ev_ebitda_median: Optional[float] = None,
) -> dict:
    """
    Compute a graduated 0-100 multi-factor composite score.

    Factor groups and maximum points
    ---------------------------------
    Quality          30 pts  (ROIC 12, Gross-margin trend 10, FCF/NI 8)
    Value            25 pts  (EV/EBIT 10, FCF yield 8, EV/EBITDA vs peers 7)
    Momentum         20 pts  (Price vs 200-SMA 8, 12-1 month return 12)
    Growth Quality   15 pts  (Rev CAGR 6, EPS leverage 5, Sloan accruals 4)
    Financial Health 10 pts  (Interest coverage 5, Debt/EBITDA 5)

    Scoring is graduated (no binary pass/fail) and sector-normalised via
    ``SECTOR_THRESHOLDS``.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    sector_ev_ebitda_median : float, optional
        Pre-computed sector-median EV/EBITDA for relative-value scoring.
        Falls back to ``SECTOR_EV_EBITDA_MEDIANS`` lookup when None.

    Returns
    -------
    dict
        Keys: ticker, quality_score, value_score, momentum_score,
        growth_score, health_score, total_score, score_breakdown,
        sector, warnings, data_quality_score.
    """
    warnings_list: list[str] = []
    breakdown: dict = {}
    n_attempted = 0
    n_fetched = 0

    out: dict = {
        "ticker": ticker,
        "quality_score": 0.0,
        "value_score": 0.0,
        "momentum_score": 0.0,
        "growth_score": 0.0,
        "health_score": 0.0,
        "total_score": 0.0,
        "score_breakdown": {},
        "sector": "Unknown",
        "warnings": [],
        "data_quality_score": 0.0,
    }

    # ── Fetch all data ────────────────────────────────────────────────────────
    try:
        d = _fetch_all(ticker)
    except Exception as exc:
        out["warnings"] = [f"Data fetch failed: {exc}"]
        return out

    info = d["info"]
    fin, q_fin = d["fin"], d["q_fin"]
    bs = d["bs"]
    cf, q_cf = d["cf"], d["q_cf"]

    sector = info.get("sector") or "Unknown"
    out["sector"] = sector
    thresholds = SECTOR_THRESHOLDS.get(sector, _DEFAULT_THRESHOLDS)

    # ── Shared computations ───────────────────────────────────────────────────
    mkt_cap = _info_val(info, "marketCap")
    ebit = _ttm_sum(q_fin, fin, "Operating Income", "Ebit", "EBIT")
    ebitda = _coalesce(
        _info_val(info, "ebitda"),
        _ttm_sum(q_fin, fin, "EBITDA", "Ebitda", "Normalized EBITDA"),
    )
    total_debt = _coalesce(
        _info_val(info, "totalDebt"),
        _latest(_get_stmt_row(bs, "Total Debt",
                              "Long Term Debt And Capital Lease Obligation")),
        0.0,
    )
    cash = _coalesce(
        _info_val(info, "totalCash"),
        _latest(_get_stmt_row(
            bs,
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Short Term Investments",
        )),
        0.0,
    )
    ev = (mkt_cap + total_debt - cash) if mkt_cap else None

    # ── ROIC (used in Quality and shared context) ─────────────────────────────
    n_attempted += 1
    roic: Optional[float] = None
    try:
        ca = _latest(_get_stmt_row(bs, "Total Current Assets", "Current Assets"))
        cl = _latest(_get_stmt_row(bs, "Total Current Liabilities",
                                    "Current Liabilities"))
        ppe = _latest(_get_stmt_row(
            bs, "Net PPE",
            "Net Property Plant And Equipment",
            "Net Property Plant Equipment",
        )) or 0.0
        st_debt = _latest(_get_stmt_row(
            bs, "Current Debt",
            "Current Debt And Capital Lease Obligation",
            "Short Term Debt",
        )) or 0.0

        if ebit and ca and cl:
            nwc = (ca - cash) - (cl - st_debt)
            ic = nwc + ppe
            if ic > 0:
                roic = ebit / ic
                n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"ROIC computation: {exc}")

    # ═══════════════════════════════════════════════════════════════════════════
    # FACTOR 1 — QUALITY  (max 30 pts)
    # ═══════════════════════════════════════════════════════════════════════════
    quality_pts = 0.0

    # ── ROIC scoring (12 pts) ─────────────────────────────────────────────────
    roic_pts: Optional[float] = None
    if roic is not None:
        pct = roic * 100
        roic_pts = 12 if pct > 20 else (9 if pct >= 15 else (6 if pct >= 10 else 3))
        quality_pts += roic_pts
    else:
        warnings_list.append("Quality – ROIC: could not compute")
    breakdown["roic_pts"] = roic_pts

    # ── Gross-margin trend (10 pts) ───────────────────────────────────────────
    n_attempted += 1
    gm_trend_pts: Optional[float] = None
    try:
        rev_row = _get_stmt_row(fin, "Total Revenue", "Revenue")
        gp_row = _get_stmt_row(fin, "Gross Profit", "Gross Income")

        if rev_row is not None and gp_row is not None:
            common = rev_row.index.intersection(gp_row.index)
            if len(common) >= 2:
                rev = rev_row[common].sort_index(ascending=False)
                gp = gp_row[common].sort_index(ascending=False)
                gm_series = (gp / rev * 100).dropna()
                if len(gm_series) >= 1:
                    gm_ttm = float(gm_series.iloc[0])
                    gm_3yr = float(gm_series.iloc[: min(3, len(gm_series))].mean())
                    delta = gm_ttm - gm_3yr
                    gm_trend_pts = (
                        10 if delta > 1.0 else (6 if delta >= -1.0 else 2)
                    )
                    quality_pts += gm_trend_pts
                    n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Quality – Gross-margin trend: {exc}")
    breakdown["gm_trend_pts"] = gm_trend_pts

    # ── FCF / Net Income ratio (8 pts) ────────────────────────────────────────
    n_attempted += 1
    fcf_ni_pts: Optional[float] = None
    try:
        fcf_ttm = _coalesce(
            _info_val(info, "freeCashflow"),
            _ttm_sum(q_cf, cf, "Free Cash Flow", "FreeCashFlow"),
        )
        ni_ttm = _ttm_sum(
            q_fin, fin,
            "Net Income", "Net Income From Continuing Operations",
            "Net Income Common Stockholders",
        )
        if fcf_ttm is not None and ni_ttm and ni_ttm > 0:
            ratio = fcf_ttm / ni_ttm
            fcf_ni_pts = (
                8 if ratio > 0.9 else (5 if ratio >= 0.7 else (2 if ratio >= 0.5 else 0))
            )
            quality_pts += fcf_ni_pts
            n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Quality – FCF/NI: {exc}")
    breakdown["fcf_ni_pts"] = fcf_ni_pts

    out["quality_score"] = quality_pts

    # ═══════════════════════════════════════════════════════════════════════════
    # FACTOR 2 — VALUE  (max 25 pts)
    # ═══════════════════════════════════════════════════════════════════════════
    value_pts = 0.0
    ev_ebit_fair = thresholds.get("ev_ebit_fair", 15)
    fcf_yield_min = thresholds.get("fcf_yield_min", 5)

    # ── EV/EBIT (10 pts) ─────────────────────────────────────────────────────
    n_attempted += 1
    ev_ebit_pts: Optional[float] = None
    try:
        if ev and ebit and ebit > 0:
            ev_ebit = ev / ebit
            # Sector-normalised breakpoints: <10x cheap, up to fair = decent,
            # up to fair*1.33 = full, above = expensive
            if ev_ebit < 10:
                ev_ebit_pts = 10
            elif ev_ebit < ev_ebit_fair:
                ev_ebit_pts = 7
            elif ev_ebit < ev_ebit_fair * 1.33:
                ev_ebit_pts = 4
            else:
                ev_ebit_pts = 1
            value_pts += ev_ebit_pts
            n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Value – EV/EBIT: {exc}")
    breakdown["ev_ebit_pts"] = ev_ebit_pts

    # ── FCF Yield (8 pts) ────────────────────────────────────────────────────
    n_attempted += 1
    fcf_yield_pts: Optional[float] = None
    try:
        fcf_ttm = _coalesce(
            _info_val(info, "freeCashflow"),
            _ttm_sum(q_cf, cf, "Free Cash Flow", "FreeCashFlow"),
        )
        if fcf_ttm is not None and mkt_cap and mkt_cap > 0:
            fcf_yield = (fcf_ttm / mkt_cap) * 100  # as %
            # Sector-normalised: 2x min = excellent, min = decent
            if fcf_yield > 2 * fcf_yield_min:
                fcf_yield_pts = 8
            elif fcf_yield >= fcf_yield_min:
                fcf_yield_pts = 5
            elif fcf_yield >= 0.6 * fcf_yield_min:
                fcf_yield_pts = 2
            else:
                fcf_yield_pts = 0
            value_pts += fcf_yield_pts
            n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Value – FCF yield: {exc}")
    breakdown["fcf_yield_pts"] = fcf_yield_pts

    # ── EV/EBITDA vs sector median (7 pts) ───────────────────────────────────
    n_attempted += 1
    ev_ebitda_pts: Optional[float] = None
    try:
        if ev and ebitda and ebitda > 0:
            ev_ebitda = ev / ebitda
            peer_median = (
                sector_ev_ebitda_median
                or SECTOR_EV_EBITDA_MEDIANS.get(sector,
                                                 SECTOR_EV_EBITDA_MEDIANS["default"])
            )
            premium = (ev_ebitda - peer_median) / peer_median
            ev_ebitda_pts = 7 if premium < -0.20 else (4 if premium <= 0.20 else 1)
            value_pts += ev_ebitda_pts
            n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Value – EV/EBITDA vs peers: {exc}")
    breakdown["ev_ebitda_pts"] = ev_ebitda_pts

    out["value_score"] = value_pts

    # ═══════════════════════════════════════════════════════════════════════════
    # FACTOR 3 — MOMENTUM  (max 20 pts)
    # ═══════════════════════════════════════════════════════════════════════════
    momentum_pts = 0.0
    n_attempted += 2  # SMA-200 and 12-1 month

    sma_pts: Optional[float] = None
    mom_pts: Optional[float] = None
    try:
        hist = d["t"].history(period="15mo", auto_adjust=True)
        close = hist["Close"].dropna()

        if len(close) >= 200:
            current_price = float(close.iloc[-1])
            sma_200 = float(close.iloc[-200:].mean())
            sma_pts = 8.0 if current_price > sma_200 else 0.0
            momentum_pts += sma_pts
            n_fetched += 1
        else:
            warnings_list.append("Momentum – SMA200: fewer than 200 days available")

        # 12-1 month return: return from 12 months ago to 1 month ago
        if len(close) >= 252:
            price_12m = float(close.iloc[-252])
            price_1m = float(close.iloc[-21])
            if price_12m > 0:
                ret_12_1 = (price_1m / price_12m - 1) * 100
                if ret_12_1 > 15:
                    mom_pts = 12.0
                elif ret_12_1 >= 5:
                    mom_pts = 8.0
                elif ret_12_1 >= -5:
                    mom_pts = 4.0
                else:
                    mom_pts = 0.0
                momentum_pts += mom_pts
                n_fetched += 1
        else:
            warnings_list.append("Momentum – 12-1 month: need 252+ days of history")

    except Exception as exc:
        warnings_list.append(f"Momentum: {exc}")

    breakdown["sma_200_pts"] = sma_pts
    breakdown["momentum_12_1_pts"] = mom_pts
    out["momentum_score"] = momentum_pts

    # ═══════════════════════════════════════════════════════════════════════════
    # FACTOR 4 — GROWTH QUALITY  (max 15 pts)
    # ═══════════════════════════════════════════════════════════════════════════
    growth_pts = 0.0

    # ── 3yr Revenue CAGR (6 pts) ──────────────────────────────────────────────
    n_attempted += 1
    rev_cagr_pts: Optional[float] = None
    rev_cagr: Optional[float] = None
    try:
        rev_row = _get_stmt_row(fin, "Total Revenue", "Revenue")
        rev_cagr = _cagr(rev_row, 3)
        if rev_cagr is not None:
            pct = rev_cagr * 100
            rev_cagr_pts = (
                6 if pct > 15 else (4 if pct >= 10 else (2 if pct >= 5 else 0))
            )
            growth_pts += rev_cagr_pts
            n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Growth – Revenue CAGR: {exc}")
    breakdown["rev_cagr_pts"] = rev_cagr_pts

    # ── EPS CAGR > Revenue CAGR — operating leverage (5 pts) ─────────────────
    n_attempted += 1
    eps_leverage_pts: Optional[float] = None
    try:
        # Prefer EPS row; fall back to Net Income as proxy
        eps_row = _get_stmt_row(
            fin, "Basic EPS", "Diluted EPS", "Basic Earnings Per Share",
        )
        proxy_row = eps_row or _get_stmt_row(
            fin, "Net Income", "Net Income From Continuing Operations",
        )
        proxy_cagr = _cagr(proxy_row, 3)
        if rev_cagr is not None and proxy_cagr is not None:
            eps_leverage_pts = 5.0 if proxy_cagr > rev_cagr else 0.0
            growth_pts += eps_leverage_pts
            n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Growth – EPS leverage: {exc}")
    breakdown["eps_leverage_pts"] = eps_leverage_pts

    # ── Sloan Accruals Ratio (4 pts) ──────────────────────────────────────────
    # Sloan = (Net Income − Operating Cash Flow) / Avg Total Assets
    # Low accruals signal that earnings are backed by real cash flows.
    n_attempted += 1
    sloan_pts: Optional[float] = None
    try:
        ni_ttm = _ttm_sum(
            q_fin, fin,
            "Net Income", "Net Income From Continuing Operations",
        )
        ocf_ttm = _coalesce(
            _info_val(info, "operatingCashflow"),
            _ttm_sum(
                q_cf, cf,
                "Operating Cash Flow", "Cash From Operations",
                "Total Cash From Operating Activities",
            ),
        )
        ta_row = _get_stmt_row(bs, "Total Assets")
        if ta_row is not None and len(ta_row) >= 2:
            avg_ta = (float(ta_row.iloc[0]) + float(ta_row.iloc[1])) / 2.0
            if ni_ttm is not None and ocf_ttm is not None and avg_ta > 0:
                # Signed ratio: high positive = NI far exceeds OCF = poor quality
                sloan = ((ni_ttm - ocf_ttm) / avg_ta) * 100  # as %
                sloan_pts = 4.0 if sloan < 2 else (2.0 if sloan < 5 else 0.0)
                growth_pts += sloan_pts
                n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Growth – Sloan accruals: {exc}")
    breakdown["sloan_pts"] = sloan_pts

    out["growth_score"] = growth_pts

    # ═══════════════════════════════════════════════════════════════════════════
    # FACTOR 5 — FINANCIAL HEALTH  (max 10 pts)
    # ═══════════════════════════════════════════════════════════════════════════
    health_pts = 0.0

    # ── Interest Coverage: EBIT / |Interest Expense| (5 pts) ─────────────────
    n_attempted += 1
    int_cov_pts: Optional[float] = None
    try:
        int_exp = abs(
            _ttm_sum(
                q_fin, fin,
                "Interest Expense",
                "Interest Expense Non Operating",
                "Net Interest Income",
            ) or 0.0
        )
        if ebit is not None:
            if int_exp > 0:
                coverage = ebit / int_exp
                int_cov_pts = (
                    5.0 if coverage > 5
                    else (3.0 if coverage >= 3 else (1.0 if coverage >= 1 else 0.0))
                )
            else:
                # Zero interest expense = effectively infinite coverage
                int_cov_pts = 5.0
            health_pts += int_cov_pts
            n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Health – Interest coverage: {exc}")
    breakdown["int_cov_pts"] = int_cov_pts

    # ── Debt / EBITDA (5 pts) ─────────────────────────────────────────────────
    n_attempted += 1
    debt_ebitda_pts: Optional[float] = None
    try:
        if ebitda and ebitda > 0:
            d_eb = total_debt / ebitda
            debt_ebitda_pts = (
                5.0 if d_eb < 1
                else (3.0 if d_eb < 2 else (1.0 if d_eb < 3 else 0.0))
            )
            health_pts += debt_ebitda_pts
            n_fetched += 1
    except Exception as exc:
        warnings_list.append(f"Health – Debt/EBITDA: {exc}")
    breakdown["debt_ebitda_pts"] = debt_ebitda_pts

    out["health_score"] = health_pts

    # ── Totals ────────────────────────────────────────────────────────────────
    total = quality_pts + value_pts + momentum_pts + growth_pts + health_pts
    out["total_score"] = min(float(total), 100.0)
    out["score_breakdown"] = breakdown
    out["warnings"] = warnings_list
    out["data_quality_score"] = round(
        n_fetched / n_attempted * 100 if n_attempted > 0 else 0.0, 1
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — REVERSE DCF VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def reverse_dcf(ticker: str, wacc: float = 0.10) -> dict:
    """
    Reverse DCF: solve for the growth rate implied by the current stock price.

    The model solves numerically for the annual FCF growth rate ``g`` such
    that a 10-year discounted cash-flow model equals the current market cap:

        Σ_{t=1}^{10} [FCF₀·(1+g)ᵗ / (1+wacc)ᵗ]
        + FCF₀·(1+g)¹⁰·(1+tgr) / [(wacc−tgr)·(1+wacc)¹⁰]
        = Market Cap

    where the terminal growth rate ``tgr = 2.5 %`` (long-run GDP proxy).

    The implied rate is then compared to the stock's historical 3-year FCF CAGR
    to determine whether the market's expectations are Reasonable, Stretched,
    or Extreme.

    Parameters
    ----------
    ticker : str
        Ticker symbol.
    wacc : float
        Weighted average cost of capital used as the discount rate (default 10 %).

    Returns
    -------
    dict
        Keys: ticker, current_fcf, market_cap, implied_growth_rate,
        historical_fcf_cagr, growth_premium, verdict.
    """
    TGR = 0.025  # terminal growth rate

    result: dict = {
        "ticker": ticker,
        "current_fcf": None,
        "market_cap": None,
        "implied_growth_rate": None,
        "historical_fcf_cagr": None,
        "growth_premium": None,
        "verdict": "Insufficient Data",
        "warnings": None,
    }

    if wacc <= TGR:
        result["warnings"] = f"WACC ({wacc:.1%}) must exceed TGR ({TGR:.1%})"
        return result

    try:
        d = _fetch_all(ticker)
        info = d["info"]
        cf = d["cf"]
        q_cf = d["q_cf"]

        # ── Market cap ───────────────────────────────────────────────────────
        mkt_cap = _info_val(info, "marketCap")
        if not mkt_cap or mkt_cap <= 0:
            result["warnings"] = "Market cap unavailable"
            return result

        # ── TTM FCF ──────────────────────────────────────────────────────────
        fcf = _coalesce(
            _info_val(info, "freeCashflow"),
            _ttm_sum(q_cf, cf, "Free Cash Flow", "FreeCashFlow"),
        )
        if fcf is None or fcf <= 0:
            result["warnings"] = "FCF not available or ≤ 0; DCF requires positive FCF"
            return result

        result["current_fcf"] = fcf
        result["market_cap"] = mkt_cap

        # ── Historical 3yr FCF CAGR ───────────────────────────────────────────
        fcf_row = _get_stmt_row(cf, "Free Cash Flow", "FreeCashFlow")
        hist_cagr = _cagr(fcf_row, 3)
        result["historical_fcf_cagr"] = hist_cagr

        # ── DCF present-value function ────────────────────────────────────────
        def _pv(g: float) -> float:
            pv = sum(
                fcf * (1 + g) ** yr / (1 + wacc) ** yr
                for yr in range(1, 11)
            )
            terminal = fcf * (1 + g) ** 10 * (1 + TGR) / ((wacc - TGR) * (1 + wacc) ** 10)
            return pv + terminal

        objective = lambda g: _pv(g) - mkt_cap

        # Search within a wide but sensible range
        g_lo, g_hi = -0.50, 3.0
        try:
            f_lo = objective(g_lo)
            f_hi = objective(g_hi)

            if f_lo * f_hi > 0:
                # Market cap outside the model range — pin to the nearest boundary
                implied_g = g_hi if f_hi < 0 else g_lo
                result["warnings"] = "DCF did not converge; implied rate pinned to boundary"
            else:
                implied_g = brentq(objective, g_lo, g_hi, xtol=1e-7, maxiter=200)

            result["implied_growth_rate"] = implied_g

        except Exception as exc:
            result["warnings"] = f"Solver failed: {exc}"
            return result

        # ── Verdict ───────────────────────────────────────────────────────────
        if hist_cagr is not None and hist_cagr > 0:
            premium = implied_g - hist_cagr
            result["growth_premium"] = premium
            if implied_g <= hist_cagr * 1.5:
                result["verdict"] = "Reasonable"
            elif implied_g <= hist_cagr * 2.0:
                result["verdict"] = "Stretched"
            else:
                result["verdict"] = "Extreme"

        elif hist_cagr is not None:
            # Negative historical CAGR; any positive implied growth is generous
            result["growth_premium"] = implied_g - hist_cagr
            result["verdict"] = (
                "Reasonable" if implied_g <= 0.05
                else ("Stretched" if implied_g <= 0.15 else "Extreme")
            )
        else:
            # No history to compare against; use absolute thresholds
            result["verdict"] = (
                "Reasonable" if implied_g <= 0.10
                else ("Stretched" if implied_g <= 0.20 else "Extreme")
            )

    except Exception as exc:
        result["warnings"] = f"reverse_dcf error: {exc}"
        logger.warning("reverse_dcf failed for %s: %s", ticker, exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def analyze_stocks(
    tickers: list[str],
    wacc: float = 0.10,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Full 3-stage institutional valuation pipeline.

    Pipeline
    --------
    1. ``magic_formula_screen`` on all tickers — filter and rank.
    2. Keep top 30 % by combined rank (minimum 3 survivors).
    3. ``multi_factor_score`` for each survivor.
    4. ``reverse_dcf`` for stocks with total_score ≥ 65.
    5. Merge results; sort by total_score descending.

    Parameters
    ----------
    tickers : list[str]
        Full list of ticker symbols to evaluate.
    wacc : float
        Discount rate passed to the reverse DCF (default 10 %).

    Returns
    -------
    tuple[pd.DataFrame, list[str]]
        ``(df, all_warnings)`` where ``df`` contains columns:
        ticker, combined_rank, total_score, quality_score, value_score,
        momentum_score, growth_score, health_score, implied_growth_rate,
        growth_premium, verdict, sector, data_quality_score, rating.
    """
    all_warnings: list[str] = []

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    _separator("STAGE 1 — Greenblatt Magic Formula Screen", len(tickers))
    screen_df = magic_formula_screen(tickers)

    if screen_df.empty:
        print("  No tickers survived the Magic Formula screen.")
        return pd.DataFrame(), all_warnings

    n_survivors = len(screen_df)
    n_keep = max(3, math.ceil(n_survivors * 0.30))
    top_df = screen_df.head(n_keep)
    top_tickers = top_df["ticker"].tolist()
    print(f"  Survived: {n_survivors}  →  top 30 % kept: {n_keep}  {top_tickers}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    _separator("STAGE 2 — Multi-Factor Scoring", len(top_tickers))

    # Pre-compute sector EV/EBITDA medians from the screened universe
    sector_medians: dict[str, float] = {}
    for _, row in top_df.iterrows():
        sec = row.get("sector", "Unknown")
        if sec not in sector_medians:
            sector_medians[sec] = SECTOR_EV_EBITDA_MEDIANS.get(
                sec, SECTOR_EV_EBITDA_MEDIANS["default"]
            )

    score_rows: list[dict] = []
    for i, ticker in enumerate(top_tickers, 1):
        print(f"  [{i:2d}/{len(top_tickers)}] Scoring {ticker:<8}", end=" … ")
        sec = top_df.loc[top_df["ticker"] == ticker, "sector"].values
        peer_median = sector_medians.get(sec[0] if len(sec) else "Unknown")
        s = multi_factor_score(ticker, sector_ev_ebitda_median=peer_median)
        print(f"total={s['total_score']:5.1f}  dq={s['data_quality_score']:.0f}%")
        all_warnings.extend(s.get("warnings", []))
        score_rows.append({
            "ticker": s["ticker"],
            "sector": s["sector"],
            "quality_score": s["quality_score"],
            "value_score": s["value_score"],
            "momentum_score": s["momentum_score"],
            "growth_score": s["growth_score"],
            "health_score": s["health_score"],
            "total_score": s["total_score"],
            "data_quality_score": s["data_quality_score"],
        })

    scores_df = pd.DataFrame(score_rows)
    merged = top_df.drop(columns=["sector"], errors="ignore").merge(
        scores_df, on="ticker", how="inner"
    )

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    dcf_tickers = merged.loc[merged["total_score"] >= 65, "ticker"].tolist()
    _separator("STAGE 3 — Reverse DCF Validation", len(dcf_tickers))

    dcf_map: dict[str, dict] = {}
    for i, ticker in enumerate(dcf_tickers, 1):
        print(f"  [{i:2d}/{len(dcf_tickers)}] DCF {ticker:<8}", end=" … ")
        dcf = reverse_dcf(ticker, wacc=wacc)
        dcf_map[ticker] = dcf
        ig = dcf.get("implied_growth_rate")
        v = dcf.get("verdict", "N/A")
        print(
            f"implied={ig*100:.1f}%  verdict={v}"
            if ig is not None
            else f"verdict={v}"
        )
        if w := dcf.get("warnings"):
            all_warnings.append(f"{ticker}/DCF: {w}")

    merged["implied_growth_rate"] = merged["ticker"].map(
        lambda t: dcf_map[t]["implied_growth_rate"] if t in dcf_map else None
    )
    merged["growth_premium"] = merged["ticker"].map(
        lambda t: dcf_map[t]["growth_premium"] if t in dcf_map else None
    )
    merged["verdict"] = merged["ticker"].map(
        lambda t: dcf_map[t]["verdict"] if t in dcf_map else None
    )
    merged["rating"] = merged["total_score"].map(_rating_label)

    # ── Final column order ────────────────────────────────────────────────────
    ordered_cols = [
        "ticker", "combined_rank", "total_score",
        "quality_score", "value_score", "momentum_score",
        "growth_score", "health_score",
        "implied_growth_rate", "growth_premium", "verdict",
        "sector", "data_quality_score", "rating",
    ]
    final_cols = [c for c in ordered_cols if c in merged.columns]
    merged = (
        merged[final_cols]
        .sort_values("total_score", ascending=False)
        .reset_index(drop=True)
    )

    return merged, all_warnings


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT / DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame) -> None:
    """
    Print a formatted terminal report of the analysis results.

    Uses ``rich`` for colored output when available; falls back to
    ``tabulate``, then plain pandas.

    Verdict colour coding : Reasonable = green, Stretched = yellow, Extreme = red
    Rating labels         : Strong Buy ≥80, Buy ≥65, Hold ≥50, Underweight ≥35, Avoid <35

    Parameters
    ----------
    df : pd.DataFrame
        Output from ``analyze_stocks``.
    """
    if df is None or df.empty:
        print("No results to display.")
        return

    _VERDICT_COLOR = {
        "Reasonable": "green",
        "Stretched": "yellow",
        "Extreme": "red",
    }
    _RATING_COLOR = {
        "Strong Buy": "bright_green",
        "Buy": "green",
        "Hold": "yellow",
        "Underweight": "orange3",
        "Avoid": "red",
    }

    def _fmt_pct(v) -> str:
        return f"{v*100:.1f}%" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "N/A"

    def _fmt_f(v, decimals=1) -> str:
        return f"{v:.{decimals}f}" if v is not None and not (isinstance(v, float) and math.isnan(v)) else "N/A"

    if HAS_RICH:
        tbl = Table(
            title="[bold cyan]Stock Valuation Report[/bold cyan]",
            show_header=True,
            header_style="bold white on dark_blue",
        )
        cols_cfg = [
            ("Ticker",     "left",  "bold"),
            ("Rank",       "right", ""),
            ("Score/100",  "right", ""),
            ("Qual/30",    "right", "dim"),
            ("Val/25",     "right", "dim"),
            ("Mom/20",     "right", "dim"),
            ("Grw/15",     "right", "dim"),
            ("Hlth/10",    "right", "dim"),
            ("Impl.Grw%",  "right", ""),
            ("Verdict",    "left",  ""),
            ("Rating",     "left",  "bold"),
            ("Sector",     "left",  "dim"),
            ("DQ%",        "right", "dim"),
        ]
        for name, justify, style in cols_cfg:
            tbl.add_column(name, justify=justify, style=style or None)

        for _, row in df.iterrows():
            verdict = row.get("verdict") or ""
            rating = row.get("rating") or ""
            vc = _VERDICT_COLOR.get(verdict, "white")
            rc = _RATING_COLOR.get(rating, "white")
            rank_val = row.get("combined_rank")
            rank_str = str(int(rank_val)) if pd.notna(rank_val) else "N/A"

            tbl.add_row(
                str(row["ticker"]),
                rank_str,
                _fmt_f(row.get("total_score")),
                _fmt_f(row.get("quality_score")),
                _fmt_f(row.get("value_score")),
                _fmt_f(row.get("momentum_score")),
                _fmt_f(row.get("growth_score")),
                _fmt_f(row.get("health_score")),
                _fmt_pct(row.get("implied_growth_rate")),
                f"[{vc}]{verdict}[/{vc}]" if verdict else "—",
                f"[{rc}]{rating}[/{rc}]" if rating else "—",
                str(row.get("sector") or "—"),
                f"{row.get('data_quality_score', 0):.0f}%",
            )

        _console.print()
        _console.print(tbl)
        _console.print(
            "\n[bold]Scoring:[/bold]  "
            "Quality [dim]/30[/dim]  Value [dim]/25[/dim]  "
            "Momentum [dim]/20[/dim]  Growth [dim]/15[/dim]  "
            "Health [dim]/10[/dim]"
        )
        _console.print(
            "[bold]Ratings:[/bold]  "
            "[bright_green]Strong Buy[/bright_green] ≥80  "
            "[green]Buy[/green] ≥65  "
            "[yellow]Hold[/yellow] ≥50  "
            "[orange3]Underweight[/orange3] ≥35  "
            "[red]Avoid[/red] <35"
        )
        _console.print(
            "[bold]Verdict:[/bold]  "
            "[green]Reasonable[/green]  "
            "[yellow]Stretched[/yellow]  "
            "[red]Extreme[/red]"
        )

    elif HAS_TABULATE:
        display = df.copy()
        if "implied_growth_rate" in display.columns:
            display["impl_grw%"] = display["implied_growth_rate"].map(_fmt_pct)
            display = display.drop(columns=["implied_growth_rate", "growth_premium"],
                                   errors="ignore")
        if "data_quality_score" in display.columns:
            display["dq%"] = display["data_quality_score"].map(lambda x: f"{x:.0f}%")
            display = display.drop(columns=["data_quality_score"], errors="ignore")
        for col in ["total_score", "quality_score", "value_score",
                    "momentum_score", "growth_score", "health_score"]:
            if col in display.columns:
                display[col] = display[col].round(1)

        print("\n" + "=" * 110)
        print("  STOCK VALUATION REPORT")
        print("=" * 110)
        print(_tabulate(display, headers="keys", tablefmt="rounded_grid",
                        showindex=False, floatfmt=".1f"))
        print("\nScoring:  Quality/30  Value/25  Momentum/20  Growth/15  Health/10")
        print("Ratings:  Strong Buy ≥80 | Buy ≥65 | Hold ≥50 | Underweight ≥35 | Avoid <35")
        print("Verdict:  Reasonable | Stretched | Extreme")

    else:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)
        print("\n" + "=" * 90)
        print("  STOCK VALUATION REPORT")
        print("=" * 90)
        print(df.to_string(index=False))
        print("\nScoring: Quality/30  Value/25  Momentum/20  Growth/15  Health/10")
        print("Ratings: Strong Buy ≥80 | Buy ≥65 | Hold ≥50 | Underweight ≥35 | Avoid <35")


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _rating_label(score: float) -> str:
    """Map a 0-100 total score to a rating string."""
    if score >= 80:
        return "Strong Buy"
    if score >= 65:
        return "Buy"
    if score >= 50:
        return "Hold"
    if score >= 35:
        return "Underweight"
    return "Avoid"


def _separator(title: str, n: int) -> None:
    """Print a section header to stdout."""
    print(f"\n{'─'*60}")
    print(f"  {title}  ({n} ticker{'s' if n != 1 else ''})")
    print(f"{'─'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "NVDA", "JPM", "JNJ", "XOM", "WMT",
    ]
    df, warnings = analyze_stocks(tickers, wacc=0.10)
    print_report(df)

    if warnings:
        print(f"\n{'─'*60}")
        print(f"  Warnings / skipped metrics ({len(warnings)} total)")
        print(f"{'─'*60}")
        for w in warnings:
            print(f"  • {w}")
