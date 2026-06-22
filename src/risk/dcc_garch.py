"""
Dynamic Conditional Correlation GARCH (Engle 2002).

Step 1: Fit univariate GARCH(p,q) to each sector return series via `arch`.
Step 2: Custom DCC layer on standardized residuals for time-varying correlation.
Crisis correlation is extracted at the percentile of average conditional
volatility defined by config.vol_stress_quantile.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.utils.logger import get_logger

try:
    from arch import arch_model as _arch_model
    _ARCH_AVAILABLE = True
except ImportError:
    _arch_model = None
    _ARCH_AVAILABLE = False

logger = get_logger(__name__)

if not _ARCH_AVAILABLE:
    logger.warning(
        "arch is not installed. DCC-GARCH fitting will be unavailable. "
        "Install with: pip install arch>=6.3.0"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class ConvergenceError(RuntimeError):
    """Raised when more than half of univariate GARCH models fail to converge."""


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DCCGARCHConfig:
    """Configuration for DCC-GARCH estimation."""

    garch_p: int = field(default=1)
    garch_q: int = field(default=1)
    dcc_alpha_init: float = field(default=0.05)
    dcc_beta_init: float = field(default=0.90)
    estimate_dcc_params: bool = field(default=True)
    distribution: str = field(default="normal")    # "normal" | "t" | "skewt"
    mean_model: str = field(default="constant")    # "constant" | "zero" | "ar"
    ar_lags: int = field(default=1)
    min_observations: int = field(default=100)
    vol_stress_quantile: float = field(default=0.95)
    max_fit_attempts: int = field(default=3)


@dataclass
class DCCGARCHResult:
    """Full output from DCCGARCHModel.fit()."""

    conditional_correlations: np.ndarray     # shape (T, N, N)
    conditional_volatilities: pd.DataFrame   # shape (T, N) — decimal units
    standardized_residuals: pd.DataFrame     # shape (T, N)
    dcc_alpha: float
    dcc_beta: float
    current_correlation: pd.DataFrame        # NxN — latest timestep
    stress_correlation: pd.DataFrame         # NxN — at vol_stress_quantile
    calm_correlation: pd.DataFrame           # NxN — at 1 - vol_stress_quantile
    sector_names: list[str]
    config: DCCGARCHConfig
    aic_per_series: dict[str, float]
    convergence_status: dict[str, bool]
    fit_warnings: list[str]


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class DCCGARCHModel:
    """
    Dynamic Conditional Correlation GARCH model (Engle 2002).

    Parameters
    ----------
    config : DCCGARCHConfig
        Runtime configuration for GARCH order, DCC parameters, and distribution.
    """

    # Mapping from config string to arch library mean model name
    _MEAN_MAP: dict[str, str] = {
        "constant": "Constant",
        "zero": "Zero",
        "ar": "AR",
    }

    def __init__(self, config: DCCGARCHConfig = DCCGARCHConfig()) -> None:
        self._config = config
        self._Q_bar: Optional[np.ndarray] = None
        self._Q_last: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_dcc_recursion(
        self,
        E: np.ndarray,
        Q_bar: np.ndarray,
        alpha: float,
        beta: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Execute the DCC recursion over the full sample.

        Parameters
        ----------
        E : np.ndarray
            Standardized residuals, shape (T, N).
        Q_bar : np.ndarray
            Unconditional covariance of residuals, shape (N, N).
        alpha : float
            DCC ARCH parameter.
        beta : float
            DCC GARCH parameter.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Q_array (T, N, N) — dynamic quasi-correlation matrices.
            R_array (T, N, N) — dynamic correlation matrices (diagonal = 1).
        """
        T, N = E.shape
        Q_array = np.empty((T, N, N), dtype=float)
        R_array = np.empty((T, N, N), dtype=float)

        one_minus = 1.0 - alpha - beta
        Q_prev = Q_bar.copy()

        for t in range(T):
            if t == 0:
                Q_t = Q_bar.copy()
            else:
                e_prev = E[t - 1]
                Q_t = (
                    one_minus * Q_bar
                    + alpha * np.outer(e_prev, e_prev)
                    + beta * Q_prev
                )

            Q_t = (Q_t + Q_t.T) * 0.5  # enforce symmetry

            q_diag = np.diag(Q_t)
            q_safe = np.where(q_diag > 1e-12, q_diag, 1e-12)
            q_inv_sqrt = 1.0 / np.sqrt(q_safe)
            R_t = Q_t * np.outer(q_inv_sqrt, q_inv_sqrt)
            np.fill_diagonal(R_t, 1.0)

            Q_array[t] = Q_t
            R_array[t] = R_t
            Q_prev = Q_t

        return Q_array, R_array

    def _neg_dcc_loglik(
        self,
        params: np.ndarray,
        E: np.ndarray,
        Q_bar: np.ndarray,
    ) -> float:
        """
        Negative DCC log-likelihood for parameter estimation.

        L = -0.5 * sum_t [ log(det(R_t)) + e_t' * R_t^{-1} * e_t ]

        Parameters
        ----------
        params : np.ndarray
            [alpha, beta] — DCC parameters to optimise.
        E : np.ndarray
            Standardized residuals (T, N).
        Q_bar : np.ndarray
            Unconditional covariance matrix (N, N).

        Returns
        -------
        float
            Negative log-likelihood value (to be minimised).
        """
        alpha, beta = float(params[0]), float(params[1])

        if alpha <= 1e-8 or beta <= 1e-8 or alpha + beta >= 0.9999:
            return 1e10

        T, N = E.shape
        ll = 0.0
        one_minus = 1.0 - alpha - beta
        Q_prev = Q_bar.copy()

        for t in range(T):
            if t == 0:
                Q_t = Q_bar.copy()
            else:
                e_prev = E[t - 1]
                Q_t = (
                    one_minus * Q_bar
                    + alpha * np.outer(e_prev, e_prev)
                    + beta * Q_prev
                )

            Q_t = (Q_t + Q_t.T) * 0.5

            q_diag = np.diag(Q_t)
            q_safe = np.where(q_diag > 1e-12, q_diag, 1e-12)
            q_inv_sqrt = 1.0 / np.sqrt(q_safe)
            R_t = Q_t * np.outer(q_inv_sqrt, q_inv_sqrt)
            np.fill_diagonal(R_t, 1.0)

            e_t = E[t]
            try:
                sign, logdet = np.linalg.slogdet(R_t)
                if sign <= 0:
                    return 1e10
                R_inv = np.linalg.inv(R_t)
                ll += logdet + float(e_t @ R_inv @ e_t)
            except np.linalg.LinAlgError:
                return 1e10

            Q_prev = Q_t

        return 0.5 * ll

    def _fit_univariate_garch(
        self,
        col: str,
        series_scaled: pd.Series,
        series_index: pd.Index,
        scale: float,
    ) -> tuple[Optional[pd.Series], Optional[pd.Series], bool, float, list[str]]:
        """
        Fit GARCH(p,q) for a single series.

        Returns
        -------
        tuple
            (cond_vol_series, std_resid_series, converged, aic, warnings)
            cond_vol_series in decimal units; None on total failure.
        """
        config = self._config
        warnings: list[str] = []
        mean_str = self._MEAN_MAP.get(config.mean_model.lower(), "Constant")

        try:
            if mean_str == "AR":
                am = _arch_model(
                    series_scaled,
                    mean=mean_str,
                    lags=config.ar_lags,
                    vol="GARCH",
                    p=config.garch_p,
                    q=config.garch_q,
                    dist=config.distribution,
                )
            else:
                am = _arch_model(
                    series_scaled,
                    mean=mean_str,
                    vol="GARCH",
                    p=config.garch_p,
                    q=config.garch_q,
                    dist=config.distribution,
                )
        except Exception as exc:
            msg = f"{col}: arch_model construction failed — {exc}"
            logger.error(msg)
            warnings.append(msg)
            return None, None, False, float("nan"), warnings

        best_fitted = None
        converged = False

        for attempt in range(config.max_fit_attempts):
            sv = None
            if attempt > 0 and best_fitted is not None:
                rng = np.random.default_rng(attempt * 17 + 3)
                sv = np.abs(
                    best_fitted.params.values
                    * (1.0 + 0.15 * rng.standard_normal(len(best_fitted.params)))
                )
            try:
                fitted = am.fit(update_freq=0, disp="off", starting_values=sv)
                if best_fitted is None or fitted.loglikelihood > best_fitted.loglikelihood:
                    best_fitted = fitted
                if int(fitted.convergence_flag) == 0:
                    converged = True
                    break
            except Exception as exc:
                logger.warning(
                    f"{col}: GARCH attempt {attempt + 1}/{config.max_fit_attempts} — {exc}"
                )

        if best_fitted is None:
            msg = f"{col}: all {config.max_fit_attempts} GARCH attempts failed."
            logger.error(msg)
            warnings.append(msg)
            return None, None, False, float("nan"), warnings

        if not converged:
            msg = (
                f"{col}: GARCH did not fully converge "
                f"(convergence_flag={int(best_fitted.convergence_flag)})."
            )
            logger.warning(msg)
            warnings.append(msg)

        sigma_arr = np.array(best_fitted.conditional_volatility, dtype=float)
        resid_arr = np.array(best_fitted.resid, dtype=float)
        sigma_safe = np.where(sigma_arr > 1e-12, sigma_arr, 1e-12)
        epsilon_arr = resid_arr / sigma_safe

        cond_vol = pd.Series(sigma_arr / scale, index=series_index, name=col)
        std_resid = pd.Series(epsilon_arr, index=series_index, name=col)

        return cond_vol, std_resid, converged, float(best_fitted.aic), warnings

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, sector_returns: pd.DataFrame) -> DCCGARCHResult:
        """
        Full DCC-GARCH estimation.

        Algorithm
        ---------
        1. For each column in sector_returns, fit GARCH(p,q) via ``arch``.
           Retry up to config.max_fit_attempts with jittered starting values.
           Extract conditional volatility and standardised residuals.
        2. Stack residuals into matrix E (T × N).
        3. Compute Q_bar = (1/T) E' E.
        4. Run DCC recursion with initial alpha/beta from config.
        5. If config.estimate_dcc_params, estimate alpha/beta via L-BFGS-B MLE.
           Re-run DCC recursion with estimated parameters.
        6. Extract stress, calm, and current correlation matrices.

        Parameters
        ----------
        sector_returns : pd.DataFrame
            Daily sector returns, shape (T, N). Columns are sector names.

        Returns
        -------
        DCCGARCHResult

        Raises
        ------
        ImportError
            If the ``arch`` package is not installed.
        ValueError
            If fewer than config.min_observations rows are provided.
        ConvergenceError
            If more than half of univariate GARCH models fail to converge.
        """
        if not _ARCH_AVAILABLE:
            raise ImportError(
                "arch is required for DCC-GARCH. Install with: pip install arch>=6.3.0"
            )

        config = self._config
        t_total = time.time()

        if len(sector_returns) < config.min_observations:
            raise ValueError(
                f"sector_returns has {len(sector_returns)} rows; "
                f"config.min_observations={config.min_observations} required."
            )

        sector_names = list(sector_returns.columns)
        N = len(sector_names)
        fit_warnings: list[str] = []
        aic_per_series: dict[str, float] = {}
        convergence_status: dict[str, bool] = {}
        cond_vols: dict[str, pd.Series] = {}
        std_resids: dict[str, pd.Series] = {}

        # ── Step 1: Univariate GARCH per sector ─────────────────────────
        SCALE = 100.0
        for col in sector_names:
            series = sector_returns[col].dropna()
            cond_vol, std_resid, converged, aic, warns = self._fit_univariate_garch(
                col, series * SCALE, series.index, SCALE
            )
            fit_warnings.extend(warns)
            convergence_status[col] = converged
            aic_per_series[col] = aic

            if cond_vol is None:
                # Total failure — constant-vol fallback
                std_val = max(float(series.std()), 1e-8) * SCALE
                cond_vols[col] = pd.Series(std_val / SCALE, index=series.index, name=col)
                std_resids[col] = pd.Series(
                    (series.values * SCALE) / std_val, index=series.index, name=col
                )
            else:
                cond_vols[col] = cond_vol
                std_resids[col] = std_resid

        # ── Convergence threshold check ──────────────────────────────────
        n_failed = sum(1 for v in convergence_status.values() if not v)
        if n_failed > N / 2:
            raise ConvergenceError(
                f"{n_failed}/{N} GARCH models failed to converge. "
                "Try: increase max_fit_attempts, use mean_model='zero', "
                "or distribution='normal'."
            )

        # ── Align all series to a common valid index ─────────────────────
        cond_vol_df = pd.DataFrame(cond_vols)
        std_resid_df = pd.DataFrame(std_resids)
        valid = cond_vol_df.notna().all(axis=1) & std_resid_df.notna().all(axis=1)
        cond_vol_df = cond_vol_df.loc[valid]
        std_resid_df = std_resid_df.loc[valid]

        T_fit = len(std_resid_df)
        if T_fit < config.min_observations:
            raise ValueError(
                f"Only {T_fit} aligned observations after GARCH fitting; "
                f"config.min_observations={config.min_observations} required."
            )

        logger.debug(
            f"Univariate GARCH done in {time.time() - t_total:.2f}s — "
            f"T={T_fit}, N={N}, failed={n_failed}"
        )

        # ── Step 2–3: E matrix and Q_bar ────────────────────────────────
        E = std_resid_df.values.astype(float)
        Q_bar = (E.T @ E) / T_fit

        # ── Steps 4–5: DCC recursion + optional MLE ─────────────────────
        dcc_alpha = float(config.dcc_alpha_init)
        dcc_beta = float(config.dcc_beta_init)

        if config.estimate_dcc_params:
            logger.info("Estimating DCC parameters via MLE (L-BFGS-B)…")
            t_opt = time.time()
            opt = minimize(
                self._neg_dcc_loglik,
                x0=np.array([dcc_alpha, dcc_beta]),
                args=(E, Q_bar),
                method="L-BFGS-B",
                bounds=[(1e-6, 0.499), (1e-6, 0.9989)],
                options={"maxiter": 200, "ftol": 1e-9},
            )
            dcc_alpha, dcc_beta = float(opt.x[0]), float(opt.x[1])

            if dcc_alpha + dcc_beta >= 1.0:
                dcc_beta = 0.999 - dcc_alpha
                msg = (
                    f"DCC alpha+beta >= 1 after MLE; "
                    f"clamped beta to {dcc_beta:.6f}."
                )
                logger.warning(msg)
                fit_warnings.append(msg)

            logger.info(
                f"DCC MLE done in {time.time() - t_opt:.2f}s — "
                f"alpha={dcc_alpha:.4f}, beta={dcc_beta:.4f}, "
                f"converged={opt.success}"
            )

        Q_array, R_array = self._run_dcc_recursion(E, Q_bar, dcc_alpha, dcc_beta)

        # Store for forecast_correlation
        self._Q_bar = Q_bar.copy()
        self._Q_last = Q_array[-1].copy()

        # ── Steps 6–7: Extract stress / calm / current correlations ─────
        avg_vol = cond_vol_df.mean(axis=1).values
        q_stress = float(np.nanpercentile(avg_vol, config.vol_stress_quantile * 100))
        q_calm = float(np.nanpercentile(avg_vol, (1.0 - config.vol_stress_quantile) * 100))

        idx_stress = int(np.argmin(np.abs(avg_vol - q_stress)))
        idx_calm = int(np.argmin(np.abs(avg_vol - q_calm)))

        stress_corr = pd.DataFrame(R_array[idx_stress], index=sector_names, columns=sector_names)
        calm_corr = pd.DataFrame(R_array[idx_calm], index=sector_names, columns=sector_names)
        current_corr = pd.DataFrame(R_array[-1], index=sector_names, columns=sector_names)

        logger.info(
            f"DCCGARCHModel.fit() complete in {time.time() - t_total:.2f}s — "
            f"alpha={dcc_alpha:.4f}, beta={dcc_beta:.4f}, "
            f"warnings={len(fit_warnings)}"
        )

        return DCCGARCHResult(
            conditional_correlations=R_array,
            conditional_volatilities=cond_vol_df,
            standardized_residuals=std_resid_df,
            dcc_alpha=dcc_alpha,
            dcc_beta=dcc_beta,
            current_correlation=current_corr,
            stress_correlation=stress_corr,
            calm_correlation=calm_corr,
            sector_names=sector_names,
            config=config,
            aic_per_series=aic_per_series,
            convergence_status=convergence_status,
            fit_warnings=fit_warnings,
        )

    def get_correlation_at_quantile(
        self,
        result: DCCGARCHResult,
        vol_quantile: float,
    ) -> pd.DataFrame:
        """
        Return the NxN correlation matrix at the timestep where average
        conditional volatility equals the specified quantile.

        Parameters
        ----------
        result : DCCGARCHResult
            Output of fit().
        vol_quantile : float
            0.95 → crisis correlation. 0.50 → median. 0.05 → tranquil.

        Returns
        -------
        pd.DataFrame
            NxN correlation matrix at the requested vol quantile timestep.
        """
        avg_vol = result.conditional_volatilities.mean(axis=1).values
        target = float(np.nanpercentile(avg_vol, vol_quantile * 100))
        idx = int(np.argmin(np.abs(avg_vol - target)))
        sectors = result.sector_names
        return pd.DataFrame(
            result.conditional_correlations[idx],
            index=sectors,
            columns=sectors,
        )

    def forecast_correlation(
        self,
        result: DCCGARCHResult,
        horizon: int,
    ) -> pd.DataFrame:
        """
        Mean-reverting DCC correlation forecast ``horizon`` days ahead.

        Q_{T+h} = (1 - (alpha+beta)^h) * Q_bar + (alpha+beta)^h * Q_T
        R_{T+h} = diag(Q_{T+h})^{-1/2} Q_{T+h} diag(Q_{T+h})^{-1/2}

        Parameters
        ----------
        result : DCCGARCHResult
            Output of fit().
        horizon : int
            Number of business days ahead to forecast.

        Returns
        -------
        pd.DataFrame
            NxN forecasted correlation matrix.

        Raises
        ------
        RuntimeError
            If fit() has not been called before forecast_correlation().
        ValueError
            If horizon is not a positive integer.
        """
        if self._Q_bar is None or self._Q_last is None:
            raise RuntimeError(
                "forecast_correlation() requires fit() to be called first."
            )
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")

        alpha = result.dcc_alpha
        beta = result.dcc_beta
        decay = (alpha + beta) ** horizon

        Q_forecast = (1.0 - decay) * self._Q_bar + decay * self._Q_last
        Q_forecast = (Q_forecast + Q_forecast.T) * 0.5

        q_diag = np.diag(Q_forecast)
        q_safe = np.where(q_diag > 1e-12, q_diag, 1e-12)
        q_inv_sqrt = 1.0 / np.sqrt(q_safe)
        R_forecast = Q_forecast * np.outer(q_inv_sqrt, q_inv_sqrt)
        np.fill_diagonal(R_forecast, 1.0)

        sectors = result.sector_names
        return pd.DataFrame(R_forecast, index=sectors, columns=sectors)


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def _smoke_test() -> None:
        import numpy as np
        import pandas as pd

        np.random.seed(0)

        n_days, n_sectors = 500, 3
        dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

        # Correlated sector returns
        cov = np.array([
            [1.50e-4, 0.60e-4, 0.30e-4],
            [0.60e-4, 1.20e-4, 0.20e-4],
            [0.30e-4, 0.20e-4, 1.80e-4],
        ])
        L = np.linalg.cholesky(cov)
        raw = (np.random.randn(n_days, n_sectors) @ L.T)
        # Add simple GARCH-like volatility clustering
        for t in range(1, n_days):
            raw[t] *= 1.0 + 0.3 * abs(raw[t - 1, 0]) * 10

        sector_returns = pd.DataFrame(
            raw, index=dates, columns=["Technology", "Financials", "Energy"]
        )

        # ── Fit with MLE disabled (faster smoke test) ──────────────────
        config = DCCGARCHConfig(
            garch_p=1,
            garch_q=1,
            estimate_dcc_params=False,   # skip MLE for speed
            dcc_alpha_init=0.05,
            dcc_beta_init=0.90,
            min_observations=60,
            vol_stress_quantile=0.95,
        )
        model = DCCGARCHModel(config)

        t0 = time.time()
        result = model.fit(sector_returns)
        elapsed = time.time() - t0
        print(f"  fit() completed in {elapsed:.2f}s")

        # ── Shape checks ────────────────────────────────────────────────
        T = len(result.conditional_volatilities)
        N = 3
        assert result.conditional_correlations.shape == (T, N, N), (
            f"conditional_correlations shape mismatch: {result.conditional_correlations.shape}"
        )
        assert result.conditional_volatilities.shape == (T, N), (
            f"conditional_volatilities shape: {result.conditional_volatilities.shape}"
        )
        assert result.standardized_residuals.shape == (T, N), (
            f"standardized_residuals shape: {result.standardized_residuals.shape}"
        )
        print(f"  Shape checks: T={T}, N={N} ✓")

        # ── Correlation matrix invariants ────────────────────────────────
        for name, mat in [
            ("current", result.current_correlation),
            ("stress",  result.stress_correlation),
            ("calm",    result.calm_correlation),
        ]:
            diag = np.diag(mat.values)
            assert np.allclose(diag, 1.0, atol=1e-9), (
                f"{name}_correlation diagonal not 1: {diag}"
            )
            # Symmetric
            assert np.allclose(mat.values, mat.values.T, atol=1e-12), (
                f"{name}_correlation not symmetric"
            )
            # Correlation values in [-1, 1]
            assert mat.values.min() >= -1.0 - 1e-9 and mat.values.max() <= 1.0 + 1e-9, (
                f"{name}_correlation out of range"
            )
        print("  current / stress / calm correlation invariants ✓")

        # ── Time-varying correlations ────────────────────────────────────
        all_corr = result.conditional_correlations
        diag_all = all_corr[:, range(N), range(N)]
        assert np.allclose(diag_all, 1.0, atol=1e-9), "Diagonal not 1 in time series"
        print("  Time-varying correlations: diagonal = 1.0 for all T ✓")

        # ── Conditional volatilities are positive ────────────────────────
        assert (result.conditional_volatilities.values > 0).all(), (
            "Conditional volatilities must be positive"
        )
        print("  Conditional volatilities > 0 ✓")

        # ── DCC params in valid range ────────────────────────────────────
        assert 0 < result.dcc_alpha < 1, f"dcc_alpha out of range: {result.dcc_alpha}"
        assert 0 < result.dcc_beta < 1, f"dcc_beta out of range: {result.dcc_beta}"
        assert result.dcc_alpha + result.dcc_beta < 1, (
            f"alpha+beta >= 1: {result.dcc_alpha + result.dcc_beta}"
        )
        print(
            f"  DCC params: alpha={result.dcc_alpha:.4f}, "
            f"beta={result.dcc_beta:.4f} ✓"
        )

        # ── get_correlation_at_quantile ─────────────────────────────────
        for q in (0.05, 0.50, 0.95):
            corr_q = model.get_correlation_at_quantile(result, q)
            assert corr_q.shape == (N, N)
            assert np.allclose(np.diag(corr_q.values), 1.0, atol=1e-9)
        print("  get_correlation_at_quantile() ✓")

        # ── forecast_correlation ────────────────────────────────────────
        for h in (1, 5, 21, 63):
            fc = model.forecast_correlation(result, horizon=h)
            assert fc.shape == (N, N), f"forecast shape wrong at h={h}"
            assert np.allclose(np.diag(fc.values), 1.0, atol=1e-9)
            assert np.allclose(fc.values, fc.values.T, atol=1e-12)
        print("  forecast_correlation() horizons 1/5/21/63d ✓")

        # Longer horizon → closer to calm (mean reversion)
        fc_short = model.forecast_correlation(result, horizon=1)
        fc_long = model.forecast_correlation(result, horizon=500)
        # Off-diagonal should converge as horizon → ∞
        off_diag_short = fc_short.values[0, 1]
        off_diag_long = fc_long.values[0, 1]
        print(
            f"  Mean reversion: off-diag at h=1: {off_diag_short:.4f}, "
            f"h=500: {off_diag_long:.4f}"
        )

        # ── forecast before fit raises RuntimeError ─────────────────────
        fresh_model = DCCGARCHModel()
        try:
            fresh_model.forecast_correlation(result, horizon=5)
            raise AssertionError("Expected RuntimeError")
        except RuntimeError:
            pass
        print("  forecast_correlation() before fit() raises RuntimeError ✓")

        # ── AIC and convergence dict ────────────────────────────────────
        assert set(result.aic_per_series.keys()) == set(result.sector_names)
        assert set(result.convergence_status.keys()) == set(result.sector_names)
        print("  aic_per_series and convergence_status keys correct ✓")

        # ── fit with MLE enabled ────────────────────────────────────────
        config_mle = DCCGARCHConfig(
            estimate_dcc_params=True,
            min_observations=60,
        )
        model_mle = DCCGARCHModel(config_mle)
        result_mle = model_mle.fit(sector_returns)
        assert 0 < result_mle.dcc_alpha < 1
        assert 0 < result_mle.dcc_beta < 1
        assert result_mle.dcc_alpha + result_mle.dcc_beta < 1
        print(
            f"  MLE-estimated DCC: alpha={result_mle.dcc_alpha:.4f}, "
            f"beta={result_mle.dcc_beta:.4f} ✓"
        )

        print("\n✓ [DCCGARCHModel] smoke test passed")

    _smoke_test()
