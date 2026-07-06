"""
Sector-neutral composite ranking algorithm for the Portfolio Builder.

This module owns the real ranking algorithm that Phase 1's cache/fetch
layer only stubbed out with interim proxies (see fetch.py's module
docstring). RankingEngine.compute_composite_score() is the intended
compute_fn target for UniverseCache.run_nightly_refresh() going forward —
wiring that swap is a follow-up, not done in this module.

Absolute constraint 3 (grep-checked, see phase report): composite weights
and turnover thresholds are config constants here. expected_return_estimate()
and Sharpe are DISPLAY-ONLY (Phase 4) and have zero references in this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FactorConfig:
    factors: tuple = ("earnings_yield", "roc", "momentum", "dcf_gap")
    # closed list — do not add a fifth factor here


@dataclass
class CompositeWeights:
    earnings_yield: float = 0.25
    roc: float = 0.25
    momentum: float = 0.25
    dcf_gap: float = 0.25
    # equal-weight starting point. Recalibrated by backtest — never
    # hand-tuned, never fed by expected_return_estimate() or Sharpe.


@dataclass
class TurnoverConfig:
    entry_percentile: float = 0.0    # TBD by backtest — placeholder, not guessed
    stay_percentile: float = 0.0     # TBD by backtest — must be <= entry_percentile
    # asymmetric bands: higher bar to enter than to remain

    def __post_init__(self) -> None:
        if self.stay_percentile > self.entry_percentile:
            raise ValueError(
                f"stay_percentile ({self.stay_percentile}) must be <= "
                f"entry_percentile ({self.entry_percentile})"
            )


@dataclass
class PointInTimeLagConfig:
    us_lag_months: int = 6
    idx_lag_months: int = 7   # TODO: validate against real observed IDX
                               # filing timestamps once available


@dataclass
class RankedStock:
    ticker: str
    sector: str
    factor_zscores: dict       # sector-neutral z-scores
    composite_score: float
    rank_tier: str             # "high" | "mid" | "low" — drives the heat color
    # NOTE: no method in this module constructs RankedStock instances yet.
    # The three methods below (compute_factor_zscore, compute_composite_score,
    # apply_point_in_time_lag) are exactly what this phase's spec and CHECK
    # ask for and hand-verify. Assembling RankedStock rows (including
    # rank_tier's percentile-bucket cutoffs, which would need their own
    # config dataclass) is left as an open question for the next phase
    # that actually consumes a ranked list end-to-end — see phase report.


class RankingEngine:
    """Sector-neutral composite scoring. See module docstring for scope."""

    def __init__(
        self,
        factor_config: FactorConfig = FactorConfig(),
        weights: CompositeWeights = CompositeWeights(),
        turnover: TurnoverConfig = TurnoverConfig(),
        lag: PointInTimeLagConfig = PointInTimeLagConfig(),
    ):
        self.factor_config = factor_config
        self.weights = weights
        self.turnover = turnover
        self.lag = lag

    def compute_factor_zscore(
        self, raw_factor_values: pd.Series, sector_map: dict
    ) -> pd.Series:
        """
        Sector-neutral z-score: for each sector, z = (x - sector_mean) /
        sector_std. Computed WITHIN each sector, never across the full
        universe — this is what stops structurally different sectors
        (e.g. banks' earnings yield) from dominating the rank.

        sector_std uses ddof=0 (population std of the sector group itself,
        not a sample estimate) — the sector's members ARE the reference
        population being scored against, not a sample drawn from some
        larger population.

        Tickers missing from sector_map get a NaN z-score (not fabricated).
        A sector with zero variance (single member, or all-equal values)
        gets z=0.0 for its members rather than inf/NaN from a divide-by-zero.
        """
        sectors = pd.Series(sector_map)
        aligned_sectors = sectors.reindex(raw_factor_values.index)

        missing = raw_factor_values.index[aligned_sectors.isna() & raw_factor_values.notna()]
        if len(missing) > 0:
            logger.warning(
                f"compute_factor_zscore: {list(missing)} missing from sector_map; "
                "z-score will be NaN for these tickers"
            )

        df = pd.DataFrame({"value": raw_factor_values, "sector": aligned_sectors})
        group_mean = df.groupby("sector")["value"].transform("mean")
        group_std = df.groupby("sector")["value"].transform(lambda s: s.std(ddof=0))

        z = (df["value"] - group_mean) / group_std

        # Degenerate (zero/undefined std) is only meaningful for a KNOWN
        # sector with no spread — e.g. a single-member sector. A missing
        # sector also produces a NaN group_std (groupby drops NaN keys),
        # but that must stay NaN, not get coerced to 0.0 alongside the
        # genuinely-degenerate case — otherwise an unmapped ticker looks
        # identical to a scored, average one.
        known_sector = aligned_sectors.notna()
        degenerate = known_sector & ((group_std == 0) | group_std.isna())
        degenerate_with_data = df.index[degenerate & df["value"].notna()]
        if len(degenerate_with_data) > 0:
            logger.warning(
                f"compute_factor_zscore: zero/undefined sector std for "
                f"{list(degenerate_with_data)}; z-score set to 0.0"
            )
        z = z.where(~degenerate, 0.0)
        z.name = raw_factor_values.name
        return z

    def compute_composite_score(
        self, zscores: pd.DataFrame, weights: CompositeWeights
    ) -> pd.Series:
        """Weighted sum of the four sector-neutral z-scores. No hard
        filters — every ticker in the input set gets a score.

        A ticker missing one factor's z-score (NaN) is treated as neutral
        (0.0) for that factor rather than dropped — matches the "no hard
        filters" requirement: partial data degrades the score, it doesn't
        remove the ticker.
        """
        factors = list(self.factor_config.factors)
        missing_cols = [f for f in factors if f not in zscores.columns]
        if missing_cols:
            logger.warning(
                f"compute_composite_score: factors {missing_cols} absent from "
                "zscores input; treated as neutral (0.0) for all tickers"
            )

        composite = pd.Series(0.0, index=zscores.index)
        for factor in factors:
            weight = getattr(weights, factor)
            if factor in zscores.columns:
                values = zscores[factor]
                nan_tickers = values.index[values.isna()]
                if len(nan_tickers) > 0:
                    logger.debug(
                        f"compute_composite_score: {factor} missing for "
                        f"{list(nan_tickers)}; treated as neutral (0.0)"
                    )
                values = values.fillna(0.0)
            else:
                values = pd.Series(0.0, index=zscores.index)
            composite = composite + values * weight

        composite.name = "composite_score"
        return composite

    def apply_point_in_time_lag(
        self, fundamentals: pd.DataFrame, as_of_date: str, market: str
    ):
        """
        Exclude any fundamental whose fiscal-year-end is within
        lag.us_lag_months / lag.idx_lag_months of as_of_date, based on
        market. Must return the EXCLUDED rows too, for the CHECK below —
        don't just silently drop them.

        DEVIATION FROM SPEC TYPE HINT: the pseudocode's signature says
        `-> pd.DataFrame`, but its own docstring requires returning the
        excluded rows too, and a single DataFrame can't carry both without
        a hacky `.attrs` sidecar. Returns `(kept, excluded)` instead —
        flagged for review, same as the Phase 1 date-literal deviation.

        fundamentals must have a 'fiscal_year_end' column (parseable as
        a date). Raises ValueError if the column is missing — that's a
        caller error, not a data-quality issue to degrade through.
        """
        if "fiscal_year_end" not in fundamentals.columns:
            raise ValueError(
                "apply_point_in_time_lag: fundamentals must have a "
                "'fiscal_year_end' column"
            )

        lag_months = self.lag.idx_lag_months if market == "IDX" else self.lag.us_lag_months
        as_of = pd.Timestamp(as_of_date)
        cutoff = as_of - pd.DateOffset(months=lag_months)
        fiscal_year_end = pd.to_datetime(fundamentals["fiscal_year_end"])

        eligible = fiscal_year_end <= cutoff
        kept = fundamentals.loc[eligible].copy()
        excluded = fundamentals.loc[~eligible].copy()

        if len(excluded) > 0:
            logger.info(
                f"apply_point_in_time_lag: excluded {len(excluded)} row(s) within "
                f"{lag_months}mo lag window (market={market}, as_of={as_of_date}, "
                f"cutoff={cutoff.date()}): {excluded.index.tolist()}"
            )

        return kept, excluded


if __name__ == "__main__":
    def _smoke_test():
        from src.portfolio_builder.ranking import (
            CompositeWeights,
            FactorConfig,
            PointInTimeLagConfig,
            RankedStock,
            RankingEngine,
            TurnoverConfig,
        )

        engine = RankingEngine()

        # ── compute_factor_zscore: hand-computable 3-stock test set ──────
        # Tech: A=10, B=20 -> mean=15, population std=5 -> zA=-1.0, zB=+1.0
        # Energy: C=30 alone -> population std=0 (degenerate) -> zC=0.0
        raw = pd.Series({"A": 10.0, "B": 20.0, "C": 30.0})
        sector_map = {"A": "Tech", "B": "Tech", "C": "Energy"}
        z = engine.compute_factor_zscore(raw, sector_map)
        assert abs(z["A"] - (-1.0)) < 1e-9, z["A"]
        assert abs(z["B"] - 1.0) < 1e-9, z["B"]
        assert z["C"] == 0.0, z["C"]
        print("✓ compute_factor_zscore: sector-neutral z-scores match hand calc, degenerate sector -> 0.0")

        # Ticker missing from sector_map -> NaN, not fabricated
        raw2 = pd.Series({"A": 10.0, "B": 20.0, "D": 99.0})
        z2 = engine.compute_factor_zscore(raw2, sector_map)  # D not in sector_map
        assert pd.isna(z2["D"])
        print("✓ compute_factor_zscore: ticker missing from sector_map -> NaN, not fabricated")

        # ── compute_composite_score: hand-computable 2-stock, equal weights ──
        # D: (1.0, 0.5, -0.5, 2.0) -> 0.25*(1.0+0.5-0.5+2.0) = 0.25*3.0 = 0.75
        # E: (-1.0, -0.5, 0.5, -2.0) -> 0.25*(-3.0) = -0.75
        zscores_df = pd.DataFrame({
            "earnings_yield": {"D": 1.0, "E": -1.0},
            "roc":            {"D": 0.5, "E": -0.5},
            "momentum":       {"D": -0.5, "E": 0.5},
            "dcf_gap":        {"D": 2.0, "E": -2.0},
        })
        composite = engine.compute_composite_score(zscores_df, CompositeWeights())
        assert abs(composite["D"] - 0.75) < 1e-9, composite["D"]
        assert abs(composite["E"] - (-0.75)) < 1e-9, composite["E"]
        print("✓ compute_composite_score: weighted sum matches hand calc")

        # Missing factor (NaN) treated as neutral (0.0), not dropped — "no hard filters"
        zscores_df2 = pd.DataFrame({
            "earnings_yield": {"F": 4.0},
            "roc": {"F": 0.0},
            "momentum": {"F": float("nan")},
            "dcf_gap": {"F": 0.0},
        })
        composite2 = engine.compute_composite_score(zscores_df2, CompositeWeights())
        assert abs(composite2["F"] - 1.0) < 1e-9, composite2["F"]  # 0.25*(4+0+0+0)
        print("✓ compute_composite_score: missing factor -> neutral 0.0, ticker still scored")

        # ── apply_point_in_time_lag ───────────────────────────────────────
        fundamentals = pd.DataFrame({
            "fiscal_year_end": {
                "G": "2025-12-31",   # US, as_of 2026-07-01, 6mo lag -> cutoff 2026-01-01 -> KEPT
                "H": "2026-03-31",   # US -> within lag window -> EXCLUDED
            }
        })
        kept, excluded = engine.apply_point_in_time_lag(fundamentals, "2026-07-01", "US")
        assert list(kept.index) == ["G"], kept.index.tolist()
        assert list(excluded.index) == ["H"], excluded.index.tolist()
        print(f"✓ apply_point_in_time_lag (US): kept={kept.index.tolist()}, "
              f"excluded={excluded.index.tolist()} (visible, not silently dropped)")

        # IDX has a longer lag (7mo) than US (6mo) for the same as_of date
        fundamentals_idx = pd.DataFrame({
            "fiscal_year_end": {"I": "2025-12-15"}
            # as_of 2026-07-01: US cutoff=2026-01-01 (would KEEP), IDX cutoff=2025-12-01 (EXCLUDES)
        })
        kept_idx, excluded_idx = engine.apply_point_in_time_lag(fundamentals_idx, "2026-07-01", "IDX")
        assert list(excluded_idx.index) == ["I"], excluded_idx.index.tolist()
        kept_us, excluded_us = engine.apply_point_in_time_lag(fundamentals_idx, "2026-07-01", "US")
        assert list(kept_us.index) == ["I"], kept_us.index.tolist()
        print("✓ apply_point_in_time_lag: IDX's longer lag excludes a fundamental US's shorter lag would keep")

        # Missing fiscal_year_end column raises, doesn't silently pass through
        try:
            engine.apply_point_in_time_lag(pd.DataFrame({"x": [1]}), "2026-07-01", "US")
            raise AssertionError("expected ValueError for missing fiscal_year_end column")
        except ValueError:
            pass
        print("✓ apply_point_in_time_lag: missing fiscal_year_end column raises ValueError")

        # ── TurnoverConfig invariant ──────────────────────────────────────
        TurnoverConfig(entry_percentile=0.8, stay_percentile=0.6)  # valid: stay <= entry
        try:
            TurnoverConfig(entry_percentile=0.5, stay_percentile=0.9)
            raise AssertionError("expected ValueError for stay_percentile > entry_percentile")
        except ValueError:
            pass
        print("✓ TurnoverConfig: stay_percentile > entry_percentile raises ValueError")

        # ── Rule 2: every new/reused name this module touches must resolve ──
        assert FactorConfig().factors == ("earnings_yield", "roc", "momentum", "dcf_gap")
        assert PointInTimeLagConfig().us_lag_months == 6
        assert RankedStock("X", "Tech", {}, 0.0, "mid").rank_tier == "mid"
        print("✓ FactorConfig / PointInTimeLagConfig / RankedStock resolve")

        print("✓ ranking.py smoke test passed")

    _smoke_test()
