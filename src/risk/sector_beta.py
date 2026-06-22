"""
Cross-sector beta matrix with dual-window stability analysis.

Beta[i][j] = sensitivity of sector i to a 1% move in sector j,
estimated from OLS regression over short (1Y) and long (3Y) rolling windows.
Stability flags surface pairs where the two windows disagree materially.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SectorBetaConfig:
    """Configuration for sector beta estimation and stability analysis."""

    short_window_days: int = field(default=252)
    long_window_days: int = field(default=756)
    min_observations: int = field(default=60)
    stability_threshold: float = field(default=0.30)
    min_sector_members: int = field(default=1)
    aggregation_method: str = field(default="equal_weight")
    # "equal_weight": simple mean of constituent returns.
    # "value_weight": market-cap weighted mean (requires market_caps argument).


@dataclass
class BetaStabilityFlag:
    """Stability flag for a single (driver, responder) sector pair."""

    sector_driver: str
    sector_responder: str
    beta_short: float
    beta_long: float
    divergence: float       # abs(beta_short - beta_long)
    is_unstable: bool       # divergence > config.stability_threshold


@dataclass
class SectorBetaResult:
    """Full output from a SectorBetaAnalyzer.compute() call."""

    beta_matrix_short: pd.DataFrame      # NxN using short_window_days
    beta_matrix_long: pd.DataFrame       # NxN using long_window_days
    beta_matrix_average: pd.DataFrame    # element-wise mean of short and long
    sector_returns: pd.DataFrame         # daily sector return series (full history)
    stability_flags: list[BetaStabilityFlag]
    sectors: list[str]
    n_unstable_pairs: int
    config: SectorBetaConfig
    computation_date: str                # ISO format date string


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class SectorBetaAnalyzer:
    """
    Build a cross-sector beta matrix with dual-window stability analysis.

    Parameters
    ----------
    config : SectorBetaConfig
        Runtime configuration for windows, thresholds, and aggregation.
    """

    def __init__(self, config: SectorBetaConfig = SectorBetaConfig()) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def build_sector_returns(
        self,
        returns: pd.DataFrame,
        sector_map: dict[str, str],
        market_caps: Optional[dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate individual ticker returns to sector-level daily return series.

        Parameters
        ----------
        returns : pd.DataFrame
            Shape (T, N). Columns are ticker strings.
        sector_map : dict[str, str]
            {ticker: sector_name}.
        market_caps : dict[str, float], optional
            Required when config.aggregation_method == "value_weight".
            Keys are ticker strings; values are market capitalisations in any
            consistent unit (absolute values, not weights).

        Returns
        -------
        pd.DataFrame
            Shape (T, S) where S = number of unique sectors retained.
            Columns are sector names. Index matches returns.index.

        Raises
        ------
        ValueError
            If aggregation_method is "value_weight" but market_caps is None.
        """
        if self._config.aggregation_method == "value_weight" and market_caps is None:
            raise ValueError(
                "market_caps must be provided when aggregation_method='value_weight'."
            )

        # Drop tickers not in sector_map
        known_tickers = [t for t in returns.columns if t in sector_map]
        dropped = set(returns.columns) - set(known_tickers)
        if dropped:
            logger.warning(
                f"build_sector_returns: {len(dropped)} ticker(s) not in sector_map "
                f"and will be dropped: {sorted(dropped)}"
            )

        if not known_tickers:
            logger.error("No tickers from returns found in sector_map. Returning empty DataFrame.")
            return pd.DataFrame(index=returns.index)

        # Group tickers by sector
        sector_to_tickers: dict[str, list[str]] = {}
        for ticker in known_tickers:
            sector = sector_map[ticker]
            sector_to_tickers.setdefault(sector, []).append(ticker)

        # Drop sectors with too few members
        valid_sectors = {
            s: tickers
            for s, tickers in sector_to_tickers.items()
            if len(tickers) >= self._config.min_sector_members
        }
        dropped_sectors = set(sector_to_tickers) - set(valid_sectors)
        if dropped_sectors:
            logger.warning(
                f"build_sector_returns: {len(dropped_sectors)} sector(s) dropped for "
                f"having fewer than {self._config.min_sector_members} member(s): "
                f"{sorted(dropped_sectors)}"
            )

        sector_series: dict[str, pd.Series] = {}
        for sector, tickers in valid_sectors.items():
            sub = returns[tickers]

            if self._config.aggregation_method == "equal_weight":
                sector_series[sector] = sub.mean(axis=1)

            else:  # value_weight
                caps = np.array([market_caps.get(t, 0.0) for t in tickers], dtype=float)
                total_cap = caps.sum()
                if total_cap <= 0.0:
                    logger.warning(
                        f"Sector '{sector}': total market cap is 0; "
                        "falling back to equal weight."
                    )
                    sector_series[sector] = sub.mean(axis=1)
                else:
                    weights = caps / total_cap
                    sector_series[sector] = (sub * weights).sum(axis=1)

        result = pd.DataFrame(sector_series, index=returns.index)
        logger.debug(
            f"build_sector_returns: {result.shape[1]} sectors built from "
            f"{len(known_tickers)} tickers over {len(result)} observations."
        )
        return result

    def _compute_beta_matrix(
        self,
        sector_returns: pd.DataFrame,
        window: int,
    ) -> pd.DataFrame:
        """
        Compute NxN beta matrix using the last `window` observations.

        beta[i][j] = cov(sector_i, sector_j) / var(sector_j).
        Diagonal = 1.0 by definition.

        Parameters
        ----------
        sector_returns : pd.DataFrame
            Full history of sector daily returns.
        window : int
            Number of trailing observations to use.

        Returns
        -------
        pd.DataFrame
            NxN DataFrame indexed and columned by sector names.
            Returns an empty DataFrame if fewer than config.min_observations
            observations are available.
        """
        available = len(sector_returns)
        n_obs = min(window, available)

        if n_obs < self._config.min_observations:
            logger.warning(
                f"_compute_beta_matrix: only {n_obs} observations available "
                f"(min_observations={self._config.min_observations}). "
                "Returning empty DataFrame."
            )
            return pd.DataFrame()

        subset = sector_returns.iloc[-n_obs:].dropna(how="all")
        sectors = list(subset.columns)
        n = len(sectors)

        data = subset.values.astype(float)
        # Covariance matrix (unbiased)
        cov = np.cov(data.T)                   # (N, N)
        variances = np.diag(cov)               # (N,)

        # Protect against zero variance (constant series)
        safe_var = np.where(variances > 0, variances, np.nan)
        # beta[i][j] = cov[i,j] / var[j]  — divide each column j by var(j)
        beta = cov / safe_var[np.newaxis, :]

        # Diagonal must be exactly 1.0; fix any floating-point drift
        np.fill_diagonal(beta, 1.0)
        # Replace NaN columns (zero-variance sectors) with 0
        beta = np.nan_to_num(beta, nan=0.0)

        return pd.DataFrame(beta, index=sectors, columns=sectors)

    def compute(
        self,
        returns: pd.DataFrame,
        sector_map: dict[str, str],
        market_caps: Optional[dict[str, float]] = None,
    ) -> SectorBetaResult:
        """
        Full computation pipeline.

        1. Aggregate ticker returns to sector-level series.
        2. Compute short-window beta matrix.
        3. Compute long-window beta matrix.
        4. Compute element-wise average matrix.
        5. Generate stability flags for all (driver, responder) sector pairs.

        Parameters
        ----------
        returns : pd.DataFrame
            Shape (T, N). Daily returns with tickers as columns.
        sector_map : dict[str, str]
            {ticker: sector_name}.
        market_caps : dict[str, float], optional
            Required for value-weighted aggregation.

        Returns
        -------
        SectorBetaResult

        Raises
        ------
        ValueError
            If fewer than config.min_observations rows are available after
            sector aggregation.
        """
        t_start = time.time()
        logger.info("SectorBetaAnalyzer.compute() started.")

        sector_returns = self.build_sector_returns(returns, sector_map, market_caps)

        if sector_returns.empty or len(sector_returns) < self._config.min_observations:
            raise ValueError(
                f"Insufficient data for beta computation: "
                f"{len(sector_returns)} rows available, "
                f"{self._config.min_observations} required."
            )

        sectors = list(sector_returns.columns)

        # ── Compute beta matrices ────────────────────────────────────────
        beta_short = self._compute_beta_matrix(
            sector_returns, self._config.short_window_days
        )
        beta_long = self._compute_beta_matrix(
            sector_returns, self._config.long_window_days
        )

        if beta_short.empty and beta_long.empty:
            raise ValueError(
                "Both short and long beta matrices are empty — insufficient data."
            )

        # Fall back to the other window if one is empty
        if beta_short.empty:
            logger.warning("Short-window beta matrix is empty; using long-window as fallback.")
            beta_short = beta_long.copy()
        if beta_long.empty:
            logger.warning("Long-window beta matrix is empty; using short-window as fallback.")
            beta_long = beta_short.copy()

        # Align indices (both matrices should share the same sectors)
        common_sectors = sorted(set(beta_short.columns) & set(beta_long.columns))
        beta_short = beta_short.loc[common_sectors, common_sectors]
        beta_long = beta_long.loc[common_sectors, common_sectors]
        beta_average = (beta_short + beta_long) / 2.0

        # ── Stability flags ──────────────────────────────────────────────
        stability_flags: list[BetaStabilityFlag] = []
        for driver in common_sectors:
            for responder in common_sectors:
                if driver == responder:
                    continue
                b_short = float(beta_short.loc[responder, driver])
                b_long = float(beta_long.loc[responder, driver])
                divergence = abs(b_short - b_long)
                is_unstable = divergence > self._config.stability_threshold
                stability_flags.append(
                    BetaStabilityFlag(
                        sector_driver=driver,
                        sector_responder=responder,
                        beta_short=b_short,
                        beta_long=b_long,
                        divergence=divergence,
                        is_unstable=is_unstable,
                    )
                )

        n_unstable = sum(1 for f in stability_flags if f.is_unstable)
        elapsed = time.time() - t_start

        logger.info(
            f"SectorBetaAnalyzer.compute() done in {elapsed:.2f}s — "
            f"{len(common_sectors)} sectors, {n_unstable} unstable pairs."
        )

        return SectorBetaResult(
            beta_matrix_short=beta_short,
            beta_matrix_long=beta_long,
            beta_matrix_average=beta_average,
            sector_returns=sector_returns[common_sectors],
            stability_flags=stability_flags,
            sectors=common_sectors,
            n_unstable_pairs=n_unstable,
            config=self._config,
            computation_date=date.today().isoformat(),
        )

    def get_implied_returns(
        self,
        result: SectorBetaResult,
        shocked_sectors: dict[str, float],
        use_window: str = "short",
    ) -> pd.Series:
        """
        Propagate sector shocks to all sectors via the beta matrix.

        For each non-shocked sector i:
            implied_return_i = sum_j( beta[i, j] * shock_j )
        For each shocked sector j:
            implied_return_j = shock_j  (override)

        Parameters
        ----------
        result : SectorBetaResult
            Output of compute().
        shocked_sectors : dict[str, float]
            {sector_name: shock_magnitude}.  E.g. {"Technology": -0.20}.
            Multiple shocked sectors are applied additively.
        use_window : str
            ``"short"`` | ``"long"`` | ``"average"``.

        Returns
        -------
        pd.Series
            Index = all sector names. Values = implied return (decimal).

        Raises
        ------
        ValueError
            If use_window is not one of the accepted values.
        """
        valid_windows = {"short", "long", "average"}
        if use_window not in valid_windows:
            raise ValueError(
                f"use_window must be one of {sorted(valid_windows)}, got '{use_window}'."
            )

        if use_window == "short":
            beta_matrix = result.beta_matrix_short
        elif use_window == "long":
            beta_matrix = result.beta_matrix_long
        else:
            beta_matrix = result.beta_matrix_average

        sectors = result.sectors
        shock_vec = pd.Series(0.0, index=sectors)
        for sector, magnitude in shocked_sectors.items():
            if sector not in shock_vec.index:
                logger.warning(
                    f"get_implied_returns: shocked sector '{sector}' not found in "
                    f"beta matrix sectors {sectors}. Skipping."
                )
            else:
                shock_vec[sector] = magnitude

        # Matrix multiply: implied[i] = sum_j( beta[i,j] * shock_j )
        # beta_matrix rows = responder (i), columns = driver (j)
        implied = beta_matrix.values @ shock_vec.values
        implied_series = pd.Series(implied, index=sectors)

        # Override shocked sectors with their exact input magnitudes
        for sector, magnitude in shocked_sectors.items():
            if sector in implied_series.index:
                implied_series[sector] = magnitude

        return implied_series

    def get_stability_report(self, result: SectorBetaResult) -> pd.DataFrame:
        """
        Return a DataFrame of all unstable sector pairs.

        Parameters
        ----------
        result : SectorBetaResult
            Output of compute().

        Returns
        -------
        pd.DataFrame
            Columns: Driver, Responder, Beta_1Y, Beta_3Y, Divergence, Warning.
            Rows correspond to pairs where divergence exceeds stability_threshold.
            Sorted by Divergence descending. Empty DataFrame if no unstable pairs.
        """
        unstable = [f for f in result.stability_flags if f.is_unstable]
        if not unstable:
            return pd.DataFrame(
                columns=["Driver", "Responder", "Beta_1Y", "Beta_3Y", "Divergence", "Warning"]
            )

        rows = [
            {
                "Driver": f.sector_driver,
                "Responder": f.sector_responder,
                "Beta_1Y": round(f.beta_short, 4),
                "Beta_3Y": round(f.beta_long, 4),
                "Divergence": round(f.divergence, 4),
                "Warning": (
                    f"Beta shifted {f.divergence:.2f} between 1Y and 3Y windows — "
                    "regime change likely."
                ),
            }
            for f in unstable
        ]

        df = pd.DataFrame(rows).sort_values("Divergence", ascending=False).reset_index(drop=True)
        return df


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def _smoke_test() -> None:
        import numpy as np
        import pandas as pd

        np.random.seed(42)

        n_days, n_tickers = 600, 9
        dates = pd.date_range("2021-01-01", periods=n_days, freq="B")

        # Three sectors, three tickers each.  Technology returns are correlated.
        tech_factor = np.random.randn(n_days) * 0.012
        fin_factor  = np.random.randn(n_days) * 0.010
        energy_factor = np.random.randn(n_days) * 0.015

        raw = {
            "AAPL":  tech_factor + np.random.randn(n_days) * 0.005,
            "MSFT":  tech_factor + np.random.randn(n_days) * 0.005,
            "GOOGL": tech_factor + np.random.randn(n_days) * 0.006,
            "JPM":   fin_factor  + np.random.randn(n_days) * 0.004,
            "BAC":   fin_factor  + np.random.randn(n_days) * 0.005,
            "WFC":   fin_factor  + np.random.randn(n_days) * 0.004,
            "XOM":   energy_factor + np.random.randn(n_days) * 0.007,
            "CVX":   energy_factor + np.random.randn(n_days) * 0.007,
            "COP":   energy_factor + np.random.randn(n_days) * 0.008,
        }
        returns = pd.DataFrame(raw, index=dates)

        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
            "JPM":  "Financials", "BAC":  "Financials", "WFC":  "Financials",
            "XOM":  "Energy",     "CVX":  "Energy",     "COP":  "Energy",
        }

        config = SectorBetaConfig(
            short_window_days=252,
            long_window_days=504,
            min_observations=60,
            stability_threshold=0.30,
            min_sector_members=1,
        )
        analyzer = SectorBetaAnalyzer(config)

        # ── build_sector_returns ────────────────────────────────────────
        sector_returns = analyzer.build_sector_returns(returns, sector_map)
        assert sector_returns.shape == (n_days, 3), (
            f"Expected (600, 3), got {sector_returns.shape}"
        )
        assert set(sector_returns.columns) == {"Technology", "Financials", "Energy"}, (
            f"Unexpected columns: {list(sector_returns.columns)}"
        )
        print(f"  build_sector_returns(): shape={sector_returns.shape} ✓")

        # ── compute ─────────────────────────────────────────────────────
        result = analyzer.compute(returns, sector_map)
        assert isinstance(result, SectorBetaResult)
        assert result.beta_matrix_short.shape == (3, 3)
        assert result.beta_matrix_long.shape == (3, 3)
        assert result.beta_matrix_average.shape == (3, 3)
        print(f"  compute(): {len(result.sectors)} sectors, "
              f"{result.n_unstable_pairs} unstable pairs ✓")

        # Diagonal must be 1.0
        for name, mat in [
            ("short", result.beta_matrix_short),
            ("long",  result.beta_matrix_long),
            ("avg",   result.beta_matrix_average),
        ]:
            diag_vals = np.diag(mat.values)
            assert np.allclose(diag_vals, 1.0, atol=1e-9), (
                f"Diagonal not 1.0 in {name} matrix: {diag_vals}"
            )
        print("  Diagonal == 1.0 for all three matrices ✓")

        # ── get_implied_returns ─────────────────────────────────────────
        for window in ("short", "long", "average"):
            implied = analyzer.get_implied_returns(
                result, {"Technology": -0.20}, use_window=window
            )
            assert isinstance(implied, pd.Series)
            assert set(implied.index) == set(result.sectors)
            assert abs(implied["Technology"] - (-0.20)) < 1e-9, (
                f"Shocked sector return wrong: {implied['Technology']}"
            )
            assert implied["Financials"] != 0.0 or implied["Energy"] != 0.0, (
                "Propagated returns should be non-zero"
            )
        print("  get_implied_returns(): shock propagation correct ✓")

        # ── Multiple shocks ─────────────────────────────────────────────
        implied_multi = analyzer.get_implied_returns(
            result,
            {"Technology": -0.20, "Energy": -0.10},
            use_window="short",
        )
        assert abs(implied_multi["Technology"] - (-0.20)) < 1e-9
        assert abs(implied_multi["Energy"] - (-0.10)) < 1e-9
        print("  get_implied_returns(): multiple shocks correct ✓")

        # ── Invalid window raises ValueError ────────────────────────────
        try:
            analyzer.get_implied_returns(result, {"Technology": -0.20}, use_window="bad")
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass
        print("  get_implied_returns(): raises ValueError on bad window ✓")

        # ── get_stability_report ────────────────────────────────────────
        report = analyzer.get_stability_report(result)
        assert isinstance(report, pd.DataFrame)
        expected_cols = {"Driver", "Responder", "Beta_1Y", "Beta_3Y", "Divergence", "Warning"}
        assert expected_cols.issubset(set(report.columns)), (
            f"Missing columns: {expected_cols - set(report.columns)}"
        )
        print(f"  get_stability_report(): {len(report)} unstable pairs ✓")

        # ── value_weight aggregation ────────────────────────────────────
        market_caps = {t: float(i + 1) * 1e9 for i, t in enumerate(raw.keys())}
        config_vw = SectorBetaConfig(aggregation_method="value_weight")
        analyzer_vw = SectorBetaAnalyzer(config_vw)
        sr_vw = analyzer_vw.build_sector_returns(returns, sector_map, market_caps=market_caps)
        assert sr_vw.shape == (n_days, 3)
        print("  value_weight aggregation ✓")

        # ── Missing market_caps raises ValueError ───────────────────────
        try:
            analyzer_vw.build_sector_returns(returns, sector_map, market_caps=None)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass
        print("  value_weight without market_caps raises ValueError ✓")

        print("\n✓ [SectorBetaAnalyzer] smoke test passed")

    _smoke_test()
