"""
Portfolio metrics for the Portfolio Builder — DISPLAY ONLY.

Two independent metric groups live here:

1. Sector exposure (HHI) + diversification rating — concentration risk,
   fusing sector spread with the Phase 3 correlation network's structure.
2. Period-matched Sharpe estimate — see below.

Both legs of the Sharpe estimate are computed over the SAME trailing
lookback_days window:

- Return leg: REALIZED historical returns of the exact constructed
  portfolio (actual input share counts) over a trailing lookback_days
  window — not a modeled/CAPM/expected return.
- Volatility leg: the existing DCC-GARCH model's OWN fitted conditional
  covariance path (conditional_correlations + conditional_volatilities),
  averaged (as variance, before the final sqrt) over that SAME trailing
  lookback_days window — reused, not refit. forecast_correlation() at any
  horizon is deliberately never used here: projecting a horizon would
  extrapolate past the disclosed, already-fitted history instead of
  summarizing it.

DEVIATION FROM THE ORIGINAL PHASE 4 SPEC: this module previously used only
the model's single LATEST day (compute_dcc_garch_volatility_current) for
the volatility leg, disclosing the resulting return/volatility period
mismatch in the UI rather than fixing it. A user-reported "impossibly high
Sharpe ratio" investigation confirmed via a controlled synthetic-data check
(not just re-reading the formula) that a current-day-only snapshot can
diverge by 2x+ from the trailing window's average whenever the current
market regime differs from that window — silently distorting Sharpe in
either direction. compute_dcc_garch_volatility_trailing() replaces it as
the volatility leg actually used by the page; compute_dcc_garch_volatility_current()
is kept (a current-moment risk reading is a legitimate, different question)
but is no longer part of the Sharpe calculation.

Isolation (absolute constraint 3): compute_realized_return(),
compute_dcc_garch_volatility_current(), compute_dcc_garch_volatility_trailing(),
and compute_sharpe() are DISPLAY-ONLY, same treatment as
expected_return_estimate() in the original spec. Phase 4's CHECK greps
ranking.py to confirm zero references to any of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DiversificationConfig:
    hhi_warning_threshold: float = 0.40  # HHI at/above this triggers a sector-concentration warning


@dataclass
class SectorExposureResult:
    sector_weights: dict   # {sector: total portfolio weight}
    hhi: float
    is_concentrated: bool  # hhi >= config.hhi_warning_threshold


def compute_sector_exposure(
    weights: pd.Series,
    sector_map: dict,
    config: DiversificationConfig = DiversificationConfig(),
) -> SectorExposureResult:
    """
    HHI = sum(sector_weight_i^2) over sectors present in the portfolio —
    the standard Herfindahl-Hirschman concentration index, applied to
    sector allocation rather than market share. `weights` need not sum to
    exactly 1.0 (a partially-invested portfolio is a valid caller concern,
    not this function's) — HHI is computed on whatever weights are given.

    A ticker with exactly zero weight (e.g. a candidate on a watchlist
    that hasn't actually been allocated any shares) contributes nothing
    to HHI either way, but its sector must not still count as "present" —
    compute_diversification_rating's N (number of sectors actually held)
    would otherwise be inflated by unfunded tickers, and the same real
    holdings would score differently depending on which zero-weight
    candidates happen to also be in the caller's ticker list. Sectors
    whose total weight is exactly 0 are dropped before returning.
    """
    unmapped = [t for t in weights.index if t not in sector_map]
    if unmapped:
        logger.warning(
            f"compute_sector_exposure: {unmapped} missing from sector_map; "
            "bucketed under 'Unknown'"
        )

    sectors = pd.Series({t: sector_map.get(t, "Unknown") for t in weights.index})
    sector_weights = weights.groupby(sectors).sum()
    sector_weights = sector_weights[sector_weights > 0]
    hhi = float((sector_weights ** 2).sum())

    return SectorExposureResult(
        sector_weights=sector_weights.to_dict(),
        hhi=hhi,
        is_concentrated=hhi >= config.hhi_warning_threshold,
    )


def compute_diversification_rating(sector_exposure: SectorExposureResult, ticker_mst) -> float:
    """
    Diversification rating on 0-100, fusing two independently-normalized
    [0,1] sub-scores via their GEOMETRIC mean — deliberately not an
    arithmetic mean, so a portfolio strong on one axis (e.g. spread across
    many sectors) but weak on the other (e.g. every ticker highly
    correlated with every other) can't get a misleadingly high blended
    score: either sub-score at 0 forces the rating to 0.

    1. Sector-spread sub-score: (1 - HHI) / (1 - 1/N), N = number of
       distinct sectors actually held. Rescales HHI's effective-N concept
       (N_eff = 1/HHI) onto [0,1]: 0 when fully concentrated in one
       sector, 1 when equally spread across all N held sectors.
       Source: Woerheide, W., & Persson, D. (1993). "An Index of
       Portfolio Diversification." Financial Services Review, 2(2),
       73-85 — proposes a Herfindahl-based index for portfolio
       diversification measurement.

    2. Correlation-density sub-score: mean Mantegna MST edge distance
       (from network.py's ticker-level MST — reused, not recomputed)
       divided by 2.0, the maximum possible Mantegna distance
       (d = sqrt(2(1-rho)), rho in [-1,1]). A "dense" network (tickers
       all tightly correlated) has short average MST edges and scores
       near 0; a network of largely independent tickers scores near 1.
       Source: Onnela, J.-P., Chakraborti, A., Kaski, K., Kertész, J., &
       Kanto, A. (2003). "Dynamics of market correlations: Taxonomy and
       portfolio analysis." Physical Review E, 68(5), 056110 — studies
       MST tree structure/length as a proxy for market correlation
       structure.

    `ticker_mst` is the networkx.Graph from network.TickerNetwork.mst —
    a graph with zero edges (e.g. a single-ticker portfolio) yields a
    correlation-density sub-score of 0.0 (no structure to measure), not
    an error, since "no diversification benefit measurable" is the
    honest answer for a one-asset portfolio.
    """
    n_sectors = len(sector_exposure.sector_weights)
    if n_sectors <= 1:
        sector_score = 0.0
    else:
        raw = (1.0 - sector_exposure.hhi) / (1.0 - 1.0 / n_sectors)
        sector_score = max(0.0, min(1.0, raw))

    edge_weights = [data["weight"] for _, _, data in ticker_mst.edges(data=True)]
    if not edge_weights:
        correlation_score = 0.0
    else:
        avg_distance = sum(edge_weights) / len(edge_weights)
        correlation_score = max(0.0, min(1.0, avg_distance / 2.0))

    rating = 100.0 * np.sqrt(sector_score * correlation_score)
    return float(rating)


@dataclass
class SharpeConfig:
    lookback_days: int = 3 * 252  # 3 years of trading days (~1 year = 252) — per explicit
    # instruction to use the past 3 years' average return and volatility,
    # not the prior 1-year window. compute_realized_return() ANNUALIZES
    # (CAGR) over this window rather than reporting the raw multi-year
    # cumulative total, so this stays an average ANNUAL rate on the same
    # scale as compute_dcc_garch_volatility_trailing's annualized
    # volatility for any lookback_days, not just the 252-day special case.
    # Annual rate (e.g. 0.045 for 4.5%) — must match the ~annual scale of
    # both compute_realized_return's annualized return and
    # compute_dcc_garch_volatility_trailing's annualized volatility.
    # Deliberately Optional with no numeric default: FRED-sourced
    # risk-free rate is broken upstream (see Phase 0 audit), so this must
    # be set explicitly as a manual constant until that's fixed.
    # compute_sharpe() raises ValueError if this is still None — it must
    # never silently default to 0.0, which would quietly overstate Sharpe.
    risk_free_rate: Optional[float] = None
    # Closed choice: "dcc_garch_trailing_average" is the only implemented/
    # allowed value. This field exists so the choice is an explicit, named,
    # auditable config value rather than an unstated assumption — not
    # because a different value actually works. compute_sharpe() raises
    # if this is changed to anything else.
    #
    # DEVIATION FROM THE ORIGINAL PHASE 4 SPEC: this was "dcc_garch_current"
    # (the model's single latest-day conditional covariance only) until a
    # user-reported "impossibly high Sharpe ratio" investigation confirmed
    # (via a controlled synthetic-data check, not just re-reading the
    # formula) that a current-day-only volatility snapshot can diverge by
    # 2x+ from the trailing lookback_days window whenever the current
    # market regime differs from that window's average — silently
    # distorting Sharpe since the return leg is a cumulative REALIZED
    # figure over that same window. "dcc_garch_trailing_average" (see
    # compute_dcc_garch_volatility_trailing) fixes the period mismatch by
    # averaging the model's own conditional variance path over the SAME
    # lookback_days window, instead of changing which model computes it.
    volatility_source: str = "dcc_garch_trailing_average"

    def __post_init__(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError(f"lookback_days must be > 0, got {self.lookback_days}")


def compute_realized_return(
    portfolio_returns: pd.Series, lookback_days: int, annualization_days: int = 252,
) -> float:
    """
    ANNUALIZED (CAGR) realized return of the ACTUAL constructed portfolio
    over the trailing lookback_days daily returns — the average ANNUAL
    rate, not the raw multi-window cumulative total: compound the window
    first, then convert via (1+cumulative)^(annualization_days/lookback_days)
    - 1. Not a modeled or CAPM-implied return: this is what the specific
    share-count portfolio the user actually built would have returned, per
    year on average.

    At lookback_days == annualization_days (the historical default,
    252 == 252, "trailing 12-month realized"), the exponent is 1 and this
    is exactly the prior cumulative-return behavior. For a multi-year
    window (e.g. lookback_days=756 for 3 years, the current default —
    "use the past 3 years' average return", not the 3-year total), this
    is what actually keeps the figure on the same annual scale as
    compute_dcc_garch_volatility_trailing's annualized volatility; a raw
    3-year cumulative total divided by a 1-year-scale volatility would
    reintroduce the exact kind of period mismatch already fixed once.

    Raises ValueError rather than silently truncating or ignoring gaps —
    a Sharpe number quietly built on a short or gappy window is worse than
    an explicit failure.
    """
    if len(portfolio_returns) < lookback_days:
        raise ValueError(
            f"compute_realized_return: only {len(portfolio_returns)} days of "
            f"returns available, need at least lookback_days={lookback_days}"
        )

    window = portfolio_returns.iloc[-lookback_days:]
    if window.isna().any():
        raise ValueError(
            "compute_realized_return: NaN value(s) in the lookback window — "
            "refusing to silently drop or fill them"
        )

    cumulative = (1.0 + window).prod() - 1.0
    years = lookback_days / annualization_days
    annualized_return = float((1.0 + cumulative) ** (1.0 / years) - 1.0)
    return annualized_return


def compute_dcc_garch_volatility_current(
    dcc_result,
    weights: pd.Series,
    annualization_days: int = 252,
) -> float:
    """
    w' Sigma w using ONLY the DCC-GARCH model's latest fitted conditional
    covariance — current_correlation (R) combined with the last row of
    conditional_volatilities (D) as Sigma = D R D. Never
    forecast_correlation() at any horizon: a projected horizon would
    compound the return/volatility period mismatch this whole module
    exists to disclose, not fix it.

    `weights` must be SECTOR weights aligned to dcc_result.sector_names —
    this DCC-GARCH engine is fit at sector level (see Phase 0 audit /
    network.py's docstring for why it was never extended to ticker
    level), so a ticker-level weights Series would silently misalign here
    without the reindex check below catching it.

    annualization_days converts the model's native daily volatility to an
    annual-scale figure so the ratio in compute_sharpe() is comparable to
    compute_realized_return's ~annual cumulative return — this is a unit
    conversion (standard sqrt-time scaling), not a business threshold, so
    it's a plain parameter with a standard default rather than a config
    dataclass field.
    """
    sectors = list(dcc_result.sector_names)

    w = weights.reindex(sectors)
    if w.isna().any():
        missing = w.index[w.isna()].tolist()
        raise ValueError(
            f"compute_dcc_garch_volatility_current: weights missing for sector(s) "
            f"{missing} (expected sector-level weights aligned to dcc_result.sector_names)"
        )

    sigma_daily = dcc_result.conditional_volatilities.iloc[-1].reindex(sectors)
    if sigma_daily.isna().any():
        missing = sigma_daily.index[sigma_daily.isna()].tolist()
        raise ValueError(
            f"compute_dcc_garch_volatility_current: conditional_volatilities "
            f"missing sector(s) {missing}"
        )

    correlation = dcc_result.current_correlation.reindex(index=sectors, columns=sectors)
    if correlation.isna().any().any():
        raise ValueError(
            "compute_dcc_garch_volatility_current: current_correlation has "
            "missing sector(s) after aligning to dcc_result.sector_names"
        )

    D = np.diag(sigma_daily.values)
    sigma_matrix_daily = D @ correlation.values @ D

    w_arr = w.values
    portfolio_variance_daily = float(w_arr @ sigma_matrix_daily @ w_arr)
    # Guard against tiny negative numerical noise from a near-singular
    # correlation matrix — a real negative variance is impossible.
    portfolio_variance_daily = max(portfolio_variance_daily, 0.0)

    daily_vol = np.sqrt(portfolio_variance_daily)
    annualized_vol = float(daily_vol * np.sqrt(annualization_days))
    return annualized_vol


def compute_dcc_garch_volatility_trailing(
    dcc_result,
    weights: pd.Series,
    lookback_days: int = 252,
    annualization_days: int = 252,
) -> float:
    """
    w' Sigma_t w averaged over the trailing lookback_days timesteps of the
    DCC-GARCH model's OWN fitted conditional covariance path (Sigma_t =
    D_t R_t D_t at each historical day t, from conditional_volatilities and
    conditional_correlations), then annualized.

    This is the period-matched replacement for
    compute_dcc_garch_volatility_current(): that function used only the
    single LATEST day's conditional covariance, which — confirmed via a
    controlled synthetic-data check (constant-vol regime: current-day and
    trailing-average agree to within noise; regime-shift scenarios: they
    can diverge by 2x+) — produces a materially different number whenever
    the current moment's volatility differs from the trailing window's
    average, silently distorting compute_sharpe() (in either direction)
    since the return leg is a cumulative realized figure over that SAME
    trailing window. Averaging the model's own daily conditional
    VARIANCES (not volatilities) over the window before the final sqrt is
    the standard way to get a representative "typical" volatility across
    a period with time-varying vol — the same principle as realized
    variance being built from squared (not linear) returns.

    lookback_days must match SharpeConfig.lookback_days (the caller's
    responsibility — see the page wiring) for the two Sharpe legs to
    actually share the same window; this function does not read
    SharpeConfig itself to avoid an import-direction dependency between
    the two dataclasses.

    Still never touches forecast_correlation() at any horizon — this
    reuses only the model's already-fitted historical conditional path,
    not a projection.
    """
    sectors = list(dcc_result.sector_names)

    w = weights.reindex(sectors)
    if w.isna().any():
        missing = w.index[w.isna()].tolist()
        raise ValueError(
            f"compute_dcc_garch_volatility_trailing: weights missing for sector(s) "
            f"{missing} (expected sector-level weights aligned to dcc_result.sector_names)"
        )

    cond_vol_df = dcc_result.conditional_volatilities.reindex(columns=sectors)
    if cond_vol_df.isna().any().any():
        missing = cond_vol_df.columns[cond_vol_df.isna().any()].tolist()
        raise ValueError(
            f"compute_dcc_garch_volatility_trailing: conditional_volatilities "
            f"missing sector(s) {missing}"
        )

    total_days = len(cond_vol_df)
    if total_days < lookback_days:
        raise ValueError(
            f"compute_dcc_garch_volatility_trailing: only {total_days} fitted day(s) "
            f"available, need at least lookback_days={lookback_days}"
        )

    # conditional_correlations is a plain (T, N, N) ndarray aligned index-for-
    # index with conditional_volatilities' rows (both built over the same T
    # timeline in DCCGARCHModel.fit() — current_correlation is documented/
    # confirmed there as exactly conditional_correlations[-1]), and its N
    # axis order is dcc_result.sector_names — the same order `sectors` was
    # just defined from, so no separate reindex is possible/needed for it.
    window_vol = cond_vol_df.iloc[-lookback_days:]
    window_corr = dcc_result.conditional_correlations[-lookback_days:]

    w_arr = w.values
    daily_variances = np.empty(lookback_days)
    for i in range(lookback_days):
        D = np.diag(window_vol.iloc[i].values)
        R = window_corr[i]
        sigma_matrix_daily = D @ R @ D
        daily_variances[i] = w_arr @ sigma_matrix_daily @ w_arr

    # Guard against tiny negative numerical noise from a near-singular
    # correlation matrix on any individual day — a real negative variance
    # is impossible.
    daily_variances = np.maximum(daily_variances, 0.0)

    avg_variance_daily = float(np.mean(daily_variances))
    daily_vol = np.sqrt(avg_variance_daily)
    annualized_vol = float(daily_vol * np.sqrt(annualization_days))
    return annualized_vol


def compute_sharpe(realized_return: float, volatility: float, config: SharpeConfig) -> float:
    """
    (realized_return - risk_free_rate) / volatility.

    Raises rather than defaulting risk_free_rate to 0.0 (would silently
    overstate Sharpe) and rather than dividing by a non-positive
    volatility (undefined/meaningless ratio).
    """
    if config.risk_free_rate is None:
        raise ValueError(
            "compute_sharpe: SharpeConfig.risk_free_rate is None — it must be set "
            "explicitly as a manual constant until FRED is fixed upstream. "
            "Refusing to silently default to 0.0."
        )
    if config.volatility_source != "dcc_garch_trailing_average":
        raise ValueError(
            f"compute_sharpe: volatility_source='{config.volatility_source}' is not "
            "supported — 'dcc_garch_trailing_average' is the only implemented source "
            "(forecast_correlation() at any horizon is deliberately not an option)."
        )
    if volatility <= 0:
        raise ValueError(f"compute_sharpe: volatility must be > 0, got {volatility}")

    logger.info(
        f"compute_sharpe: lookback_days={config.lookback_days}, "
        f"volatility_source={config.volatility_source}, "
        f"risk_free_rate={config.risk_free_rate}, "
        f"realized_return={realized_return:.6f}, volatility={volatility:.6f}"
    )
    return (realized_return - config.risk_free_rate) / volatility


def render_sharpe_methodology_disclosure() -> None:
    """
    Two-part progressive disclosure, not one dense string: a short badge
    that expands to the full caveat. Renders nothing else and computes
    nothing — purely a display component, isolated from ranking.py and the
    composite score same as the functions above.

    RENAMED from render_mixed_period_disclosure(): the return and
    volatility legs used to come from two different reference periods
    (disclosed here as a caveat); compute_dcc_garch_volatility_trailing()
    fixed that mismatch, so the old name/caveat would now be actively
    misleading (there's no period mismatch left to disclose). What's
    still worth disclosing: volatility is a GARCH-MODELED estimate over
    that window, not the portfolio's own raw realized standard deviation
    — a real, smaller methodological note, not the same caveat renamed.
    """
    import streamlit as st

    with st.expander("Sharpe methodology", expanded=False):
        st.caption(
            "Return and volatility are both computed over the same "
            "trailing lookback window. Volatility is the DCC-GARCH "
            "model's own conditional-variance path over that window, not "
            "the portfolio's raw realized standard deviation. Not a "
            "forward estimate."
        )


if __name__ == "__main__":
    def _smoke_test():
        import networkx as nx

        from src.portfolio_builder.metrics import (
            DiversificationConfig,
            SectorExposureResult,
            SharpeConfig,
            compute_dcc_garch_volatility_current,
            compute_dcc_garch_volatility_trailing,
            compute_diversification_rating,
            compute_realized_return,
            compute_sector_exposure,
            compute_sharpe,
            render_sharpe_methodology_disclosure,
        )

        # ── compute_sector_exposure: hand-computable 3-ticker, 2-sector set ──
        # Tech = A(0.3)+B(0.3) = 0.6, Energy = C(0.4) -> HHI = 0.6^2+0.4^2 = 0.52
        weights = pd.Series({"A": 0.3, "B": 0.3, "C": 0.4})
        sector_map = {"A": "Tech", "B": "Tech", "C": "Energy"}
        exposure = compute_sector_exposure(weights, sector_map)
        assert abs(exposure.sector_weights["Tech"] - 0.6) < 1e-9
        assert abs(exposure.sector_weights["Energy"] - 0.4) < 1e-9
        assert abs(exposure.hhi - 0.52) < 1e-9, exposure.hhi
        assert exposure.is_concentrated is True, "0.52 >= default 0.40 threshold"
        print("✓ compute_sector_exposure: HHI matches hand calc, concentration flag correct")

        # Ticker missing from sector_map -> bucketed under Unknown, not dropped
        weights_gap = pd.Series({"A": 0.5, "Z": 0.5})
        exposure_gap = compute_sector_exposure(weights_gap, {"A": "Tech"})
        assert "Unknown" in exposure_gap.sector_weights
        assert abs(exposure_gap.sector_weights["Unknown"] - 0.5) < 1e-9
        print("✓ compute_sector_exposure: unmapped ticker bucketed under Unknown, not dropped")

        # Below the concentration threshold
        even_weights = pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25})
        even_sectors = {"A": "S1", "B": "S2", "C": "S3", "D": "S4"}
        even_exposure = compute_sector_exposure(even_weights, even_sectors)
        assert abs(even_exposure.hhi - 0.25) < 1e-9  # 4 * 0.25^2 = 0.25
        assert even_exposure.is_concentrated is False
        print("✓ compute_sector_exposure: evenly-spread portfolio is not flagged concentrated")

        # Regression: an unfunded (zero-weight) ticker must not inflate N
        # sectors "actually held" (independent review caught this: the same
        # real 50/50 Tech/Health holding scored differently in the app
        # depending on whether a zero-share watchlist ticker in another
        # sector happened to also be present).
        funded_weights = pd.Series({"A": 0.5, "B": 0.5, "Z": 0.0})
        funded_sectors = {"A": "Tech", "B": "Health", "Z": "Energy"}
        funded_exposure = compute_sector_exposure(funded_weights, funded_sectors)
        assert "Energy" not in funded_exposure.sector_weights, (
            "a zero-weight ticker's sector must not appear as 'held'"
        )
        assert set(funded_exposure.sector_weights.keys()) == {"Tech", "Health"}
        assert abs(funded_exposure.hhi - 0.5) < 1e-9  # 0.5^2 + 0.5^2, Energy excluded
        print("✓ compute_sector_exposure: zero-weight ticker's sector excluded from 'sectors held'")

        # ── compute_diversification_rating: hand-computable fusion ──────────
        # sector_score = (1-0.52)/(1-1/2) = 0.48/0.5 = 0.96
        # MST with edges 0.8 and 1.2 -> avg distance 1.0 -> correlation_score = 0.5
        # rating = 100 * sqrt(0.96 * 0.5) = 100 * sqrt(0.48)
        mst = nx.Graph()
        mst.add_edge("A", "B", weight=0.8)
        mst.add_edge("B", "C", weight=1.2)
        rating = compute_diversification_rating(exposure, mst)
        expected_rating = 100.0 * np.sqrt(0.96 * 0.5)
        assert abs(rating - expected_rating) < 1e-9, (rating, expected_rating)
        print("✓ compute_diversification_rating: matches hand-computed geometric-mean fusion")

        # Single sector -> sector_score forced to 0 -> rating 0 regardless of correlation structure
        one_sector_exposure = SectorExposureResult(sector_weights={"Tech": 1.0}, hhi=1.0, is_concentrated=True)
        rating_one_sector = compute_diversification_rating(one_sector_exposure, mst)
        assert rating_one_sector == 0.0
        print("✓ compute_diversification_rating: single-sector portfolio -> rating 0.0")

        # No MST edges (single-ticker portfolio) -> correlation_score 0 -> rating 0, not an error
        empty_mst = nx.Graph()
        empty_mst.add_node("A")
        rating_no_edges = compute_diversification_rating(exposure, empty_mst)
        assert rating_no_edges == 0.0
        print("✓ compute_diversification_rating: no MST edges -> rating 0.0, not an error")

        # DiversificationConfig threshold is overridable
        strict_config = DiversificationConfig(hhi_warning_threshold=0.90)
        lenient_exposure = compute_sector_exposure(weights, sector_map, strict_config)
        assert lenient_exposure.is_concentrated is False, "0.52 < overridden 0.90 threshold"
        print("✓ DiversificationConfig.hhi_warning_threshold is overridable")

        # ── compute_realized_return: hand-computable constant-return series ──
        # 252 days of a constant 0.0005 daily return -> cumulative = 1.0005^252 - 1
        daily_r = 0.0005
        dates = pd.bdate_range("2025-01-01", periods=260)
        returns = pd.Series(daily_r, index=dates)
        realized = compute_realized_return(returns, lookback_days=252)
        expected = (1.0 + daily_r) ** 252 - 1.0
        assert abs(realized - expected) < 1e-12, (realized, expected)
        print("✓ compute_realized_return: matches hand-computed compounded return")

        # 3-year window (lookback_days=756, the current SharpeConfig default)
        # with the SAME constant daily rate -> the ANNUALIZED figure must
        # equal the exact same per-year rate as the 1-year case above (since
        # a constant daily return has an invariant average annual rate
        # regardless of window length), NOT the much larger raw 3-year
        # cumulative total -- this is the actual "average, not total" fix.
        dates_3yr = pd.bdate_range("2023-01-01", periods=760)
        returns_3yr = pd.Series(daily_r, index=dates_3yr)
        realized_3yr_annualized = compute_realized_return(returns_3yr, lookback_days=756)
        cumulative_3yr_total = (1.0 + daily_r) ** 756 - 1.0
        assert abs(realized_3yr_annualized - expected) < 1e-9, (realized_3yr_annualized, expected)
        assert realized_3yr_annualized < cumulative_3yr_total, (
            "annualized 3yr return must be far smaller than the raw 3yr "
            "cumulative total for a positive-return series, or the "
            "annualization isn't actually happening"
        )
        print(
            f"✓ compute_realized_return: 3-year window annualizes to the same "
            f"per-year rate ({realized_3yr_annualized:.6f}) as the 1-year case, "
            f"not the raw 3yr total ({cumulative_3yr_total:.6f})"
        )

        # Too few observations -> raises
        try:
            compute_realized_return(returns.iloc[:100], lookback_days=252)
            raise AssertionError("expected ValueError for insufficient history")
        except ValueError:
            pass
        print("✓ compute_realized_return: insufficient history raises ValueError")

        # NaN in window -> raises
        gappy = returns.copy()
        gappy.iloc[-5] = float("nan")
        try:
            compute_realized_return(gappy, lookback_days=252)
            raise AssertionError("expected ValueError for NaN in window")
        except ValueError:
            pass
        print("✓ compute_realized_return: NaN in lookback window raises ValueError")

        # ── compute_dcc_garch_volatility_current: independent recomputation ──
        sectors = ["Tech", "Energy"]
        cond_vol = pd.DataFrame({"Tech": [0.010, 0.012], "Energy": [0.008, 0.009]})
        current_corr = pd.DataFrame([[1.0, 0.3], [0.3, 1.0]], index=sectors, columns=sectors)
        weights = pd.Series({"Tech": 0.6, "Energy": 0.4})

        class _FakeDCCResult:
            sector_names = sectors
            conditional_volatilities = cond_vol
            current_correlation = current_corr

        vol = compute_dcc_garch_volatility_current(_FakeDCCResult(), weights, annualization_days=252)

        # Independent recomputation (same formula, written fresh, not reusing the function's internals)
        sigma_last = cond_vol.iloc[-1][sectors].values
        D = np.diag(sigma_last)
        R = current_corr.loc[sectors, sectors].values
        sigma_daily_matrix = D @ R @ D
        w = weights.reindex(sectors).values
        expected_var_daily = w @ sigma_daily_matrix @ w
        expected_vol = np.sqrt(expected_var_daily) * np.sqrt(252)
        assert abs(vol - expected_vol) < 1e-12, (vol, expected_vol)
        print("✓ compute_dcc_garch_volatility_current: matches independent w'Sigma w recomputation")

        # Ticker-level (mismatched) weights -> raises rather than silently misaligning
        bad_weights = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        try:
            compute_dcc_garch_volatility_current(_FakeDCCResult(), bad_weights)
            raise AssertionError("expected ValueError for sector/ticker weight mismatch")
        except ValueError:
            pass
        print("✓ compute_dcc_garch_volatility_current: mismatched weight index raises ValueError")

        # ── compute_dcc_garch_volatility_trailing: period-matched replacement ──
        # 5 days of fitted history; lookback_days=3 -- the first 2 days carry
        # a deliberately huge volatility so that, if the windowing were
        # broken and they leaked in, the result would be wildly larger than
        # what's computed below. Correlation is held constant across days so
        # the hand-computable part is just each day's w'Sigma w from its own
        # (Tech, Energy) volatility pair.
        sectors_t = ["Tech", "Energy"]
        cond_vol_trailing = pd.DataFrame({
            "Tech":   [5.0, 5.0, 0.010, 0.020, 0.010],
            "Energy": [5.0, 5.0, 0.008, 0.008, 0.012],
        })
        corr_matrix_t = np.array([[1.0, 0.3], [0.3, 1.0]])
        corr_path = np.array([corr_matrix_t] * 5)  # same correlation every day
        weights_t = pd.Series({"Tech": 0.6, "Energy": 0.4})

        class _FakeTrailingDCCResult:
            sector_names = sectors_t
            conditional_volatilities = cond_vol_trailing
            conditional_correlations = corr_path

        trailing_vol = compute_dcc_garch_volatility_trailing(
            _FakeTrailingDCCResult(), weights_t, lookback_days=3, annualization_days=252
        )

        # Independent recomputation over ONLY the last 3 rows (index 2, 3, 4)
        # -- freshly written, not reusing the function's own loop.
        w_t = weights_t.reindex(sectors_t).values
        window = cond_vol_trailing.iloc[-3:]
        daily_vars = []
        for i in range(3):
            D_i = np.diag(window.iloc[i][sectors_t].values)
            var_i = w_t @ (D_i @ corr_matrix_t @ D_i) @ w_t
            daily_vars.append(var_i)
        expected_trailing_vol = float(np.sqrt(np.mean(daily_vars)) * np.sqrt(252))
        assert abs(trailing_vol - expected_trailing_vol) < 1e-12, (trailing_vol, expected_trailing_vol)
        print("✓ compute_dcc_garch_volatility_trailing: matches independent trailing-average w'Sigma w recomputation")

        # Confirm the windowing actually EXCLUDES the huge-volatility days
        # outside lookback_days=3 -- if they'd leaked in, this would be
        # orders of magnitude larger than the ~annual-scale result above.
        assert trailing_vol < 1.0, (
            f"trailing_vol={trailing_vol} suggests the excluded huge-vol days leaked into the window"
        )
        print("✓ compute_dcc_garch_volatility_trailing: correctly excludes days outside the lookback window")

        # Fewer fitted days than lookback_days -> raises, doesn't silently truncate
        try:
            compute_dcc_garch_volatility_trailing(_FakeTrailingDCCResult(), weights_t, lookback_days=10)
            raise AssertionError("expected ValueError for insufficient fitted history")
        except ValueError:
            pass
        print("✓ compute_dcc_garch_volatility_trailing: fewer fitted days than lookback_days raises ValueError")

        # Ticker-level (mismatched) weights -> raises rather than silently misaligning
        try:
            compute_dcc_garch_volatility_trailing(_FakeTrailingDCCResult(), bad_weights, lookback_days=3)
            raise AssertionError("expected ValueError for sector/ticker weight mismatch")
        except ValueError:
            pass
        print("✓ compute_dcc_garch_volatility_trailing: mismatched weight index raises ValueError")

        # ── compute_sharpe ────────────────────────────────────────────────
        config = SharpeConfig(risk_free_rate=0.04)
        sharpe = compute_sharpe(0.12, 0.18, config)
        assert abs(sharpe - (0.12 - 0.04) / 0.18) < 1e-12
        print("✓ compute_sharpe: matches hand calc")

        # risk_free_rate=None -> raises, never silently defaults to 0.0
        try:
            compute_sharpe(0.12, 0.18, SharpeConfig(risk_free_rate=None))
            raise AssertionError("expected ValueError for risk_free_rate=None")
        except ValueError:
            pass
        print("✓ compute_sharpe: risk_free_rate=None raises ValueError, never defaults to 0.0")

        # non-positive volatility -> raises
        try:
            compute_sharpe(0.12, 0.0, SharpeConfig(risk_free_rate=0.04))
            raise AssertionError("expected ValueError for volatility <= 0")
        except ValueError:
            pass
        print("✓ compute_sharpe: non-positive volatility raises ValueError")

        # unsupported volatility_source -> raises
        bad_config = SharpeConfig(risk_free_rate=0.04)
        bad_config.volatility_source = "forecast_correlation_h21"
        try:
            compute_sharpe(0.12, 0.18, bad_config)
            raise AssertionError("expected ValueError for unsupported volatility_source")
        except ValueError:
            pass
        print("✓ compute_sharpe: non-'dcc_garch_trailing_average' volatility_source raises ValueError")

        # ── SharpeConfig validation ───────────────────────────────────────
        try:
            SharpeConfig(lookback_days=0)
            raise AssertionError("expected ValueError for lookback_days<=0")
        except ValueError:
            pass
        print("✓ SharpeConfig: lookback_days<=0 raises ValueError")

        # ── Rule 2: reused/new names resolve ──────────────────────────────
        assert callable(render_sharpe_methodology_disclosure)
        print("✓ render_sharpe_methodology_disclosure resolves")

        print("✓ metrics.py smoke test passed")

    _smoke_test()
