"""
Macro sensitivity matrix estimator for the Leontief contagion model.

Estimates S[sector][macro_variable] via OLS or Ridge regression.
S is the first-order mapping from macro shocks to sector returns,
before any inter-sector contagion propagation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Optional dependency guards ────────────────────────────────────────────────

try:
    import sqlite3 as _sqlite3
    _SQLITE3_AVAILABLE = True
except ImportError:
    _sqlite3 = None
    _SQLITE3_AVAILABLE = False
    logger.warning("sqlite3 unavailable — S matrix persistence disabled")

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
    import fredapi as _fredapi
    _FREDAPI_AVAILABLE = True
except ImportError:
    _fredapi = None
    _FREDAPI_AVAILABLE = False


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class MacroSensitivityConfig:
    """
    Configuration for MacroSensitivityEstimator.

    Parameters
    ----------
    estimation_window_days : int
        Trailing days of data used for estimation. Default 1260 (5Y).
    return_frequency : str
        Frequency of returns used in regression. "W" weekly.
    min_observations : int
        Minimum aligned weekly observations required. Default 52 (1Y).
    regularization : str
        "ridge" (default) or "ols".
    ridge_alpha : float
        Regularization strength for Ridge. Default 0.01.
    rolling : bool
        Whether to use rolling-window estimation. Default True.
    rolling_window_days : int
        Rolling window length in days. Default 504 (2Y).
    max_age_days : int
        Re-estimate if cached S matrix is older. Default 30 days.
    db_path : str
        SQLite path for persisting coefficients.
    standardize_macro : bool
        Standardize macro variables before regression so coefficients
        are comparable across variables. Default True.
    """

    estimation_window_days: int = field(default=1260)
    return_frequency: str = field(default="W")
    min_observations: int = field(default=52)
    regularization: str = field(default="ridge")
    ridge_alpha: float = field(default=0.01)
    rolling: bool = field(default=True)
    rolling_window_days: int = field(default=504)
    max_age_days: int = field(default=30)
    db_path: str = field(default="data/macro_params.db")
    standardize_macro: bool = field(default=True)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class SensitivityResult:
    """
    Output of MacroSensitivityEstimator.estimate().

    Attributes
    ----------
    S : pd.DataFrame
        (N_sectors, N_macro) regression coefficients.
    S_standardized : pd.DataFrame
        S with standardized macro inputs (zero mean, unit var).
    r_squared : pd.DataFrame
        (N_sectors, 1) in-sample R² per sector regression.
    t_statistics : pd.DataFrame
        (N_sectors, N_macro) t-statistics for each coefficient.
    n_observations : int
        Number of aligned weekly observations used.
    estimation_date : str
        ISO timestamp when estimation was performed.
    sector_names : list[str]
    macro_variable_names : list[str]
    config : MacroSensitivityConfig
    warnings : list[str]
    """

    S: pd.DataFrame
    S_standardized: pd.DataFrame
    r_squared: pd.DataFrame
    t_statistics: pd.DataFrame
    n_observations: int
    estimation_date: str
    sector_names: list[str]
    macro_variable_names: list[str]
    config: MacroSensitivityConfig
    warnings: list[str]


# ── Estimator ─────────────────────────────────────────────────────────────────

class MacroSensitivityEstimator:
    """
    Estimate the macro sensitivity matrix S via OLS or Ridge regression.

    Parameters
    ----------
    config : MacroSensitivityConfig
        Runtime configuration.
    """

    def __init__(
        self, config: MacroSensitivityConfig = MacroSensitivityConfig()
    ) -> None:
        self._config = config
        self._scaler: Optional[StandardScaler] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def estimate(
        self,
        sector_returns: pd.DataFrame,
        macro_data: pd.DataFrame,
    ) -> SensitivityResult:
        """
        Estimate S matrix via OLS or Ridge regression.

        For each sector i:
            r_i = alpha_i + S[i,:] · X + epsilon_i
        where X is the aligned macro variable matrix (T, N_macro).

        Algorithm
        ---------
        1. Align sector_returns and macro_data to common weekly dates.
        2. Slice to last config.estimation_window_days.
        3. If config.standardize_macro: standardize X columns (zero mean, unit var).
        4. If config.regularization == "ridge":
               model = Ridge(alpha=config.ridge_alpha, fit_intercept=True)
           elif "ols":
               model = LinearRegression(fit_intercept=True)
        5. Fit per sector. Store coefficients in S matrix.
        6. Compute R² and t-statistics.
        7. Save to SQLite via _save_to_db().

        Parameters
        ----------
        sector_returns : pd.DataFrame
            Daily or weekly sector return series, shape (T, N_sectors).
        macro_data : pd.DataFrame
            Macro variables already transformed and resampled to weekly,
            shape (T, N_macro). From MacroDataResult.aligned_weekly.

        Returns
        -------
        SensitivityResult

        Raises
        ------
        ValueError
            If fewer than config.min_observations aligned rows available.
        """
        t0 = time.time()
        warnings: list[str] = []

        # Step 1: Resample sector_returns to weekly if it looks daily/business-daily
        if len(sector_returns) > 1:
            _total_span = (sector_returns.index[-1] - sector_returns.index[0]).days
            _avg_gap = _total_span / max(1, len(sector_returns) - 1)
            if _avg_gap < 4:  # ≤3 day average gap → daily or business-daily
                sr_weekly = sector_returns.resample("W").sum()
            else:
                sr_weekly = sector_returns.copy()
        else:
            sr_weekly = sector_returns.copy()

        # Align to common weekly dates (direct intersection first)
        common_idx = sr_weekly.index.intersection(macro_data.index)
        if len(common_idx) < self._config.min_observations:
            # Fallback: outer-join + ffill bridges weekly vs monthly macro variables
            merged = pd.concat(
                [sr_weekly, macro_data], axis=1, join="outer"
            ).ffill().dropna()
            if len(merged) < self._config.min_observations:
                raise ValueError(
                    f"Only {len(merged)} aligned observations available; "
                    f"need at least {self._config.min_observations}. "
                    "Check that sector_returns and macro_data overlap in time."
                )
            sector_cols = [c for c in merged.columns if c in sector_returns.columns]
            macro_cols = [c for c in merged.columns if c in macro_data.columns]
            y_all = merged[sector_cols]
            X_raw = merged[macro_cols]
        else:
            y_all = sr_weekly.loc[common_idx]
            X_raw = macro_data.loc[common_idx]

        # Step 2: Slice to estimation window
        window_obs = int(
            self._config.estimation_window_days
            * len(y_all) / max(1, (y_all.index[-1] - y_all.index[0]).days)
        ) if len(y_all) > 1 else len(y_all)
        window_obs = max(self._config.min_observations, min(window_obs, len(y_all)))
        y_all = y_all.iloc[-window_obs:]
        X_raw = X_raw.iloc[-window_obs:]

        # Drop any fully NaN columns
        X_raw = X_raw.dropna(axis=1, how="all")
        y_all = y_all.dropna(axis=1, how="all")

        # Final aligned arrays
        idx = y_all.index.intersection(X_raw.index)
        y_all = y_all.loc[idx]
        X_raw_aligned = X_raw.loc[idx]

        n_obs, n_macro = X_raw_aligned.shape
        sector_names = list(y_all.columns)
        macro_names = list(X_raw_aligned.columns)

        if n_obs < self._config.min_observations:
            raise ValueError(
                f"Only {n_obs} aligned observations after windowing; "
                f"need {self._config.min_observations}."
            )

        # Step 3: Standardize macro variables
        self._scaler = StandardScaler()
        X_std = self._scaler.fit_transform(X_raw_aligned.values)
        X_arr = X_std if self._config.standardize_macro else X_raw_aligned.values

        # Step 4-6: Fit per sector
        coef_matrix = np.zeros((len(sector_names), n_macro))
        coef_std_matrix = np.zeros((len(sector_names), n_macro))
        r2_values = np.zeros(len(sector_names))
        t_stat_matrix = np.zeros((len(sector_names), n_macro))

        for i, sector in enumerate(sector_names):
            y = y_all[sector].values
            try:
                if self._config.regularization == "ridge":
                    model = Ridge(
                        alpha=self._config.ridge_alpha, fit_intercept=True
                    )
                else:
                    model = LinearRegression(fit_intercept=True)

                model.fit(X_arr, y)
                coef = model.coef_
                intercept = model.intercept_
                r2 = float(model.score(X_arr, y))

                # Coefficients in standardized space
                coef_std_matrix[i] = coef

                # Un-standardize coefficients for S (raw macro units)
                if self._config.standardize_macro:
                    scales = self._scaler.scale_
                    coef_raw = coef / np.where(scales > 0, scales, 1.0)
                else:
                    coef_raw = coef.copy()

                coef_matrix[i] = coef_raw
                r2_values[i] = r2

                # T-statistics (on raw X for interpretability)
                t_stats = self._compute_t_statistics(
                    X_raw_aligned.values, y, coef_raw, float(intercept)
                )
                t_stat_matrix[i] = t_stats

            except Exception as exc:
                warnings.append(
                    f"Regression failed for sector '{sector}': {exc} — using zeros"
                )
                logger.warning(f"MacroSensitivity: sector '{sector}' regression failed: {exc}")

        S = pd.DataFrame(coef_matrix, index=sector_names, columns=macro_names)
        S_std = pd.DataFrame(coef_std_matrix, index=sector_names, columns=macro_names)
        r_squared = pd.DataFrame(
            {"r_squared": r2_values}, index=sector_names
        )
        t_statistics = pd.DataFrame(
            t_stat_matrix, index=sector_names, columns=macro_names
        )

        estimation_date = datetime.now().isoformat()
        result = SensitivityResult(
            S=S,
            S_standardized=S_std,
            r_squared=r_squared,
            t_statistics=t_statistics,
            n_observations=n_obs,
            estimation_date=estimation_date,
            sector_names=sector_names,
            macro_variable_names=macro_names,
            config=self._config,
            warnings=warnings,
        )

        elapsed = time.time() - t0
        logger.info(
            f"MacroSensitivityEstimator.estimate() done in {elapsed:.2f}s — "
            f"sectors={len(sector_names)}, macro_vars={len(macro_names)}, "
            f"n_obs={n_obs}"
        )

        self._save_to_db(result)
        return result

    def get_initial_distress(
        self,
        result: SensitivityResult,
        macro_shock: pd.Series,
    ) -> pd.Series:
        """
        Compute h_initial = S · macro_shock.

        This is the first-order impact on each sector before any
        inter-sector contagion propagation.

        Parameters
        ----------
        result : SensitivityResult
            Output of estimate().
        macro_shock : pd.Series
            Index = macro variable names matching result.macro_variable_names.
            Values = shock magnitude in the same units as the macro variables
            (pct for DXY/IDR/commodities; bps for rate variables; PMI points).

        Returns
        -------
        pd.Series
            Index = sector names. Values = initial impact in decimal return units.
            Positive = sector gains; negative = sector loses from this shock.
        """
        S = result.S
        aligned_shock = macro_shock.reindex(S.columns).fillna(0.0)
        h_initial = S @ aligned_shock
        return h_initial

    def _save_to_db(self, result: SensitivityResult) -> None:
        """
        Persist S matrix coefficients to SQLite (config.db_path).

        Table: macro_sensitivity (estimation_date, sector, macro_var, coefficient, r_squared).
        Creates the database and table if they do not exist.
        Stores only scalar coefficients — no DataFrames in the DB.

        Parameters
        ----------
        result : SensitivityResult
        """
        if not _SQLITE3_AVAILABLE:
            logger.warning("sqlite3 unavailable — S matrix not persisted")
            return
        db_path = self._config.db_path
        try:
            db_dir = os.path.dirname(os.path.abspath(db_path))
            os.makedirs(db_dir, exist_ok=True)
            conn = _sqlite3.connect(db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS macro_sensitivity (
                    estimation_date TEXT,
                    sector TEXT,
                    macro_var TEXT,
                    coefficient REAL,
                    r_squared REAL,
                    PRIMARY KEY (estimation_date, sector, macro_var)
                )
            """)
            rows = []
            for sector in result.S.index:
                r2 = float(result.r_squared.loc[sector, "r_squared"]) if sector in result.r_squared.index else 0.0
                for macro_var in result.S.columns:
                    rows.append((
                        result.estimation_date,
                        sector,
                        macro_var,
                        float(result.S.loc[sector, macro_var]),
                        r2,
                    ))
            conn.executemany(
                "INSERT OR REPLACE INTO macro_sensitivity VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()
            conn.close()
            logger.debug(
                f"MacroSensitivity: saved {len(rows)} coefficients to {db_path}"
            )
        except Exception as exc:
            logger.error(f"_save_to_db() failed: {exc}")

    def load_from_db(
        self, max_age_days: int = None
    ) -> Optional[SensitivityResult]:
        """
        Load the most recent S matrix from SQLite if within max_age_days.

        Parameters
        ----------
        max_age_days : int, optional
            Defaults to config.max_age_days if not specified.

        Returns
        -------
        SensitivityResult or None
            None if no record found, record too old, or sqlite3 unavailable.
        """
        if not _SQLITE3_AVAILABLE:
            return None
        max_age = max_age_days if max_age_days is not None else self._config.max_age_days
        db_path = self._config.db_path
        if not os.path.exists(db_path):
            return None
        try:
            conn = _sqlite3.connect(db_path)
            cursor = conn.execute(
                "SELECT MAX(estimation_date) FROM macro_sensitivity"
            )
            row = cursor.fetchone()
            if row is None or row[0] is None:
                conn.close()
                return None
            last_date_str = row[0]
            try:
                last_date = datetime.fromisoformat(last_date_str)
            except ValueError:
                conn.close()
                return None
            age_days = (datetime.now() - last_date).days
            if age_days > max_age:
                logger.info(
                    f"Cached S matrix is {age_days} days old (max {max_age}) — will re-estimate"
                )
                conn.close()
                return None

            # Load coefficients
            df = pd.read_sql(
                "SELECT sector, macro_var, coefficient, r_squared "
                "FROM macro_sensitivity WHERE estimation_date = ?",
                conn,
                params=(last_date_str,),
            )
            conn.close()
            if df.empty:
                return None

            S = df.pivot(index="sector", columns="macro_var", values="coefficient")
            r2_df = df.drop_duplicates("sector").set_index("sector")[["r_squared"]]

            sector_names = list(S.index)
            macro_names = list(S.columns)

            n_rows = len(sector_names)
            result = SensitivityResult(
                S=S,
                S_standardized=pd.DataFrame(
                    np.zeros((n_rows, len(macro_names))),
                    index=sector_names,
                    columns=macro_names,
                ),
                r_squared=r2_df,
                t_statistics=pd.DataFrame(
                    np.zeros((n_rows, len(macro_names))),
                    index=sector_names,
                    columns=macro_names,
                ),
                n_observations=0,
                estimation_date=last_date_str,
                sector_names=sector_names,
                macro_variable_names=macro_names,
                config=self._config,
                warnings=["Loaded from cache — t-statistics and n_obs not available"],
            )
            logger.info(
                f"MacroSensitivity: loaded S matrix from DB "
                f"({age_days} days old, {len(sector_names)} sectors)"
            )
            return result
        except Exception as exc:
            logger.warning(f"load_from_db() failed: {exc}")
            return None

    def _compute_t_statistics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coef: np.ndarray,
        intercept: float,
    ) -> np.ndarray:
        """
        Compute t-statistics for Ridge/OLS coefficients.

        Uses the OLS sandwich estimator (approximate for Ridge).
        Returns an array of t-statistics for each feature coefficient,
        not including the intercept term.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_obs, n_features) — raw (not standardized) units.
        y : np.ndarray
            Target vector (n_obs,).
        coef : np.ndarray
            Regression coefficients (n_features,).
        intercept : float
            Intercept term.

        Returns
        -------
        np.ndarray
            T-statistics shape (n_features,). Returns zeros on numerical failure.
        """
        n, p = X.shape
        if n <= p + 1:
            return np.zeros(p)
        try:
            y_pred = X @ coef + intercept
            residuals = y - y_pred
            mse = float(np.sum(residuals ** 2) / max(n - p - 1, 1))
            X_aug = np.column_stack([np.ones(n), X])
            XtX = X_aug.T @ X_aug
            XtX_inv = np.linalg.pinv(XtX)
            var_coef = np.diag(XtX_inv) * mse
            se = np.sqrt(np.maximum(var_coef, 0.0))
            all_coef = np.concatenate([[intercept], coef])
            with np.errstate(divide="ignore", invalid="ignore"):
                t_all = np.where(se > 0, all_coef / se, 0.0)
            return t_all[1:]  # exclude intercept t-stat
        except (np.linalg.LinAlgError, Exception) as exc:
            logger.debug(f"_compute_t_statistics failed: {exc}")
            return np.zeros(p)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _smoke_test() -> None:
        rng = np.random.default_rng(42)
        T = 300

        dates_daily = pd.date_range("2019-01-07", periods=T, freq="W")
        sector_names = ["Technology", "Financials", "Energy"]
        macro_names = ["DXY", "VIX", "US_10Y"]

        # Synthetic sector returns: driven by macro factors + noise
        macro_factor = rng.normal(0, 0.01, (T, 3))
        S_true = np.array([
            [-0.5,  0.3, -0.2],
            [ 0.4, -0.2, -0.3],
            [ 0.1,  0.0,  0.5],
        ])
        sector_ret = macro_factor @ S_true.T + rng.normal(0, 0.005, (T, 3))

        sector_returns = pd.DataFrame(
            sector_ret, index=dates_daily, columns=sector_names
        )
        macro_data = pd.DataFrame(
            macro_factor, index=dates_daily, columns=macro_names
        )

        config = MacroSensitivityConfig(
            estimation_window_days=3000,
            min_observations=20,
            regularization="ridge",
            ridge_alpha=0.01,
            standardize_macro=True,
            db_path="/tmp/test_macro_sensitivity.db",
        )
        estimator = MacroSensitivityEstimator(config)
        result = estimator.estimate(sector_returns, macro_data)

        assert result.S.shape == (3, 3), f"Expected (3,3), got {result.S.shape}"
        assert result.r_squared.shape[0] == 3
        assert len(result.sector_names) == 3
        assert len(result.macro_variable_names) == 3
        print(f"  S matrix shape: {result.S.shape}")
        print(f"  R-squared: {result.r_squared['r_squared'].tolist()}")
        print(f"  estimation_date: {result.estimation_date}")

        # get_initial_distress
        shock = pd.Series({"DXY": 0.05, "VIX": 10.0, "US_10Y": 0.50})
        h_initial = estimator.get_initial_distress(result, shock)
        assert isinstance(h_initial, pd.Series)
        assert set(h_initial.index) == set(sector_names)
        print(f"  h_initial: {h_initial.to_dict()}")

        # load_from_db
        loaded = estimator.load_from_db()
        if loaded is not None:
            assert loaded.S.shape == result.S.shape
            print(f"  load_from_db(): S shape={loaded.S.shape}")

        # Clean up
        import os
        if os.path.exists("/tmp/test_macro_sensitivity.db"):
            os.remove("/tmp/test_macro_sensitivity.db")

        print("\nmacro_sensitivity smoke test PASSED")

    _smoke_test()
