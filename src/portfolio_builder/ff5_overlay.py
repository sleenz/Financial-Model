"""
Rolling 3-factor (Mkt-RF, SMB, CMA) regression overlay for the Portfolio
Builder — HML and RMW are dropped per the build spec.

Strictly post-hoc / display: this module takes a constructed portfolio's
returns as an INPUT and produces regression diagnostics as an OUTPUT. It
is never called from ranking.py and never feeds the composite score or
the backtest objective (grep-checked at the end of this module's phase
report, per absolute constraint 3).

Reuse (Phase 0 audit): factor return data is fetched via the existing
src.factors.fama_french.get_factor_data() (yfinance/pandas_datareader
with synthetic-data fallback) rather than reimplementing that fetch —
the 5-factor pull already includes Mkt-RF, SMB, CMA (and RF), this
module just narrows to the 3 it needs and never uses HML/RMW. No
existing method in fama_french.py already exposes this exact factor
subset (FamaFrenchAnalyzer.rolling_betas hardcodes either the 3-factor
{Mkt-RF,SMB,HML} or 5-factor set), so the rolling-window regression
itself is new here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.factors.fama_french import get_factor_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FF3CMAConfig:
    factors: tuple = ("Mkt-RF", "SMB", "CMA")   # HML, RMW dropped per spec — closed list
    rolling_window_days: int = 60
    risk_free_col: str = "RF"
    min_window_observations: int = 20            # below this, a rolling window's regression is unreliable


class FF3CMAOverlay:
    """Rolling 3-factor (Mkt-RF, SMB, CMA) regression. Post-hoc only —
    takes portfolio_returns as an input, never writes back into ranking.py."""

    def __init__(self, config: FF3CMAConfig = FF3CMAConfig()):
        self.config = config

    def fetch_factor_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Reuses fama_french.get_factor_data (5-factor pull, includes RF),
        then narrows to just this overlay's 3 factors + RF — does not
        reimplement the fetch/fallback logic."""
        full = get_factor_data(start_date, end_date, model="5")
        missing = [f for f in self.config.factors if f not in full.columns]
        if missing:
            raise ValueError(f"fetch_factor_data: factor(s) {missing} not present in fetched data")
        cols = list(self.config.factors)
        if self.config.risk_free_col in full.columns:
            cols.append(self.config.risk_free_col)
        return full[cols]

    def compute_rolling_exposures(
        self,
        portfolio_returns: pd.Series,
        factor_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Rolling OLS over config.rolling_window_days:
            portfolio_excess_return = alpha + sum(beta_f * factor_f) + residual

        factor_data defaults to a fresh fetch_factor_data() call spanning
        portfolio_returns' date range if not supplied.

        Returns a DataFrame indexed by date (the last date of each window)
        with columns: alpha, one column per config.factors, r_squared,
        n_observations. Windows with fewer than config.min_window_observations
        valid (non-NaN) rows are skipped and logged, not silently fit on
        too little data.
        """
        if factor_data is None:
            start = portfolio_returns.index[0].strftime("%Y-%m-%d")
            end = portfolio_returns.index[-1].strftime("%Y-%m-%d")
            factor_data = self.fetch_factor_data(start, end)

        factors = list(self.config.factors)
        missing = [f for f in factors if f not in factor_data.columns]
        if missing:
            raise ValueError(f"compute_rolling_exposures: factor_data missing {missing}")

        common_dates = portfolio_returns.index.intersection(factor_data.index)
        if len(common_dates) == 0:
            raise ValueError(
                "compute_rolling_exposures: no overlapping dates between "
                "portfolio_returns and factor_data"
            )

        returns = portfolio_returns.loc[common_dates].sort_index()
        factors_df = factor_data.loc[common_dates].sort_index()

        if self.config.risk_free_col in factors_df.columns:
            excess_returns = returns - factors_df[self.config.risk_free_col]
        else:
            logger.warning(
                f"compute_rolling_exposures: risk_free_col '{self.config.risk_free_col}' "
                "not in factor_data; using raw returns (not excess returns)"
            )
            excess_returns = returns

        window = self.config.rolling_window_days
        if len(excess_returns) < window:
            logger.warning(
                f"compute_rolling_exposures: only {len(excess_returns)} observations, "
                f"fewer than rolling_window_days={window}; no windows can be fit"
            )
            return pd.DataFrame(columns=["alpha", *factors, "r_squared", "n_observations"])

        rows = []
        for i in range(window, len(excess_returns) + 1):
            y = excess_returns.iloc[i - window:i].values
            X = factors_df[factors].iloc[i - window:i].values
            valid = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
            n_valid = int(valid.sum())
            if n_valid < self.config.min_window_observations:
                logger.debug(
                    f"compute_rolling_exposures: window ending "
                    f"{excess_returns.index[i - 1]} has only {n_valid} valid "
                    f"observations (< min_window_observations="
                    f"{self.config.min_window_observations}); skipped"
                )
                continue

            X_valid = np.column_stack([np.ones(n_valid), X[valid]])
            y_valid = y[valid]
            try:
                beta, *_ = np.linalg.lstsq(X_valid, y_valid, rcond=None)
            except np.linalg.LinAlgError as exc:
                logger.warning(
                    f"compute_rolling_exposures: regression failed at window ending "
                    f"{excess_returns.index[i - 1]}: {exc}"
                )
                continue

            y_pred = X_valid @ beta
            residuals = y_valid - y_pred
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((y_valid - y_valid.mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

            rows.append({
                "date": excess_returns.index[i - 1],
                "alpha": float(beta[0]),
                **{f: float(beta[j + 1]) for j, f in enumerate(factors)},
                "r_squared": r_squared,
                "n_observations": n_valid,
            })

        if not rows:
            logger.warning("compute_rolling_exposures: no window had enough observations to fit")
            return pd.DataFrame(columns=["alpha", *factors, "r_squared", "n_observations"])

        return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    def _smoke_test():
        from src.portfolio_builder.ff5_overlay import FF3CMAConfig, FF3CMAOverlay, get_factor_data

        # ── Synthetic, network-free regression: exact recovery ───────────
        # Construct factor data and a portfolio return series that is an
        # EXACT known linear combination of the 3 factors + RF, with zero
        # residual noise, so the rolling OLS must recover the true alpha
        # and betas (up to floating-point precision) — this is the
        # regression-code equivalent of Phase 2's "hand-computable" check.
        rng = np.random.default_rng(42)
        n_days = 40
        dates = pd.bdate_range("2026-01-01", periods=n_days)

        mkt_rf = rng.normal(0.0004, 0.01, n_days)
        smb = rng.normal(0.0001, 0.005, n_days)
        cma = rng.normal(0.0001, 0.004, n_days)
        rf = np.full(n_days, 0.00005)
        factor_data = pd.DataFrame(
            {"Mkt-RF": mkt_rf, "SMB": smb, "CMA": cma, "RF": rf}, index=dates
        )

        true_alpha, true_b_mkt, true_b_smb, true_b_cma = 0.0002, 1.1, 0.3, -0.2
        excess = true_alpha + true_b_mkt * mkt_rf + true_b_smb * smb + true_b_cma * cma
        portfolio_returns = pd.Series(excess + rf, index=dates)  # returns = RF + excess

        config = FF3CMAConfig(rolling_window_days=20, min_window_observations=15)
        overlay = FF3CMAOverlay(config)
        result = overlay.compute_rolling_exposures(portfolio_returns, factor_data=factor_data)

        assert not result.empty, "expected at least one fitted window"
        last = result.iloc[-1]
        assert abs(last["alpha"] - true_alpha) < 1e-8, last["alpha"]
        assert abs(last["Mkt-RF"] - true_b_mkt) < 1e-8, last["Mkt-RF"]
        assert abs(last["SMB"] - true_b_smb) < 1e-8, last["SMB"]
        assert abs(last["CMA"] - true_b_cma) < 1e-8, last["CMA"]
        assert last["r_squared"] > 0.999, last["r_squared"]
        assert last["n_observations"] == config.rolling_window_days
        print("✓ compute_rolling_exposures: exact recovery of known alpha/betas from noiseless synthetic factors")

        # ── HML/RMW are never referenced by config or output ─────────────
        assert config.factors == ("Mkt-RF", "SMB", "CMA")
        assert "HML" not in result.columns and "RMW" not in result.columns
        print("✓ FF3CMAConfig: HML/RMW dropped, only Mkt-RF/SMB/CMA present in output")

        # ── Too few total observations -> empty result, not a crash ──────
        short_returns = portfolio_returns.iloc[:10]
        short_factors = factor_data.iloc[:10]
        empty_result = overlay.compute_rolling_exposures(short_returns, factor_data=short_factors)
        assert empty_result.empty
        print("✓ compute_rolling_exposures: fewer observations than the window -> empty result, no crash")

        # ── Missing factor column raises clearly ──────────────────────────
        try:
            overlay.compute_rolling_exposures(portfolio_returns, factor_data=factor_data.drop(columns=["CMA"]))
            raise AssertionError("expected ValueError for missing CMA column")
        except ValueError:
            pass
        print("✓ compute_rolling_exposures: missing factor column raises ValueError")

        # ── Rule 2: every new/reused name this module touches must resolve ──
        assert callable(get_factor_data)
        print("✓ get_factor_data (reused from fama_french) resolves")

        print("✓ ff5_overlay.py smoke test passed")

    _smoke_test()
