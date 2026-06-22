"""
Student-t copula for joint tail dependence between sector returns.

Captures empirical "crash-together" behaviour that linear beta misses.
Supports Gaussian copula as an alternative. Provides conditional simulation,
tail dependence coefficients, and joint exceedance probability estimates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from scipy.stats import kendalltau, norm, t as t_dist

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CopulaConfig:
    """Configuration for copula fitting and simulation."""

    copula_type: str = field(default="t")              # "t" | "gaussian"
    degrees_of_freedom: float = field(default=4.0)
    estimate_df: bool = field(default=True)
    df_search_bounds: tuple = field(default=(2.01, 30.0))
    n_simulation_paths: int = field(default=10000)
    tail_quantile: float = field(default=0.05)
    random_seed: int = field(default=42)
    marginal_transform: str = field(default="empirical")
    # "empirical": rank / (T+1).  "normal": parametric normal CDF.
    correlation_estimator: str = field(default="kendall")
    # "kendall": Kendall tau → sin(pi/2 * tau).  "pearson": direct Pearson.


@dataclass
class CopulaResult:
    """Full output from StudentTCopula.fit()."""

    copula_type: str
    correlation_matrix: pd.DataFrame            # NxN fitted copula correlation
    degrees_of_freedom: float
    upper_tail_dependence: pd.DataFrame         # NxN lambda_U coefficients
    lower_tail_dependence: pd.DataFrame         # NxN lambda_L coefficients (crashes)
    simulated_uniforms: np.ndarray              # (n_paths, N) copula draws in [0, 1]
    simulated_returns: pd.DataFrame             # (n_paths, N) back-transformed returns
    joint_exceedance_probs: pd.DataFrame        # pairwise P(both < tail_quantile)
    config: CopulaConfig


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class StudentTCopula:
    """
    Student-t copula for sector return joint tail dependence.

    Supports Gaussian copula as a limiting case (copula_type="gaussian").

    Parameters
    ----------
    config : CopulaConfig
        Runtime configuration for copula type, simulation, and estimation.
    """

    def __init__(self, config: CopulaConfig = CopulaConfig()) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        """
        Return the nearest positive-definite correlation matrix to A.

        Uses eigendecomposition: clips negative eigenvalues to a small
        positive constant, then re-normalises the diagonal to 1 (Higham 2002).

        Parameters
        ----------
        A : np.ndarray
            Square symmetric matrix that may not be positive definite.

        Returns
        -------
        np.ndarray
            Nearest positive-definite correlation matrix (unit diagonal).
        """
        B = (A + A.T) / 2.0                        # enforce exact symmetry
        eigvals, eigvecs = np.linalg.eigh(B)
        eigvals = np.maximum(eigvals, 1e-8)         # clip non-positive eigenvalues
        B_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        # Re-normalise to correlation matrix (unit diagonal)
        d = np.sqrt(np.diag(B_pd))
        d_safe = np.where(d > 0, d, 1.0)
        B_corr = B_pd / np.outer(d_safe, d_safe)
        np.fill_diagonal(B_corr, 1.0)
        return B_corr

    def _to_uniform_margins(
        self,
        returns_arr: np.ndarray,
        T: int,
    ) -> np.ndarray:
        """
        Transform raw returns to uniform [0, 1] margins.

        Parameters
        ----------
        returns_arr : np.ndarray
            Shape (T, N). Raw return observations.
        T : int
            Number of time observations.

        Returns
        -------
        np.ndarray
            Shape (T, N). Values in (0, 1).
        """
        if self._config.marginal_transform == "empirical":
            # Rank-based: u_it = rank(r_it) / (T + 1)
            ranks = np.apply_along_axis(
                lambda col: np.argsort(np.argsort(col)).astype(float) + 1.0,
                axis=0,
                arr=returns_arr,
            )
            return ranks / (T + 1.0)
        else:  # normal
            means = returns_arr.mean(axis=0)
            stds = returns_arr.std(axis=0, ddof=1)
            stds = np.where(stds > 0, stds, 1.0)
            return norm.cdf((returns_arr - means) / stds)

    def _estimate_correlation(self, U_arr: np.ndarray) -> np.ndarray:
        """
        Estimate NxN copula correlation matrix from uniform-margin data.

        Parameters
        ----------
        U_arr : np.ndarray
            Shape (T, N). Uniform-margin data.

        Returns
        -------
        np.ndarray
            NxN positive-definite correlation matrix.
        """
        N = U_arr.shape[1]

        if self._config.correlation_estimator == "kendall":
            tau_mat = np.eye(N)
            for i in range(N):
                for j in range(i + 1, N):
                    tau_val, _ = kendalltau(U_arr[:, i], U_arr[:, j])
                    tau_ij = float(tau_val) if not np.isnan(tau_val) else 0.0
                    tau_mat[i, j] = tau_ij
                    tau_mat[j, i] = tau_ij

            rho_mat = np.sin(np.pi / 2.0 * tau_mat)
            np.fill_diagonal(rho_mat, 1.0)

            # Ensure positive definiteness
            try:
                np.linalg.cholesky(rho_mat)
            except np.linalg.LinAlgError:
                logger.warning(
                    "Kendall-derived correlation matrix is not PD; "
                    "applying nearest-PD projection."
                )
                rho_mat = self._nearest_positive_definite(rho_mat)

            return rho_mat

        else:  # pearson — direct on uniform data
            rho_mat = np.corrcoef(U_arr.T)
            try:
                np.linalg.cholesky(rho_mat)
            except np.linalg.LinAlgError:
                rho_mat = self._nearest_positive_definite(rho_mat)
            return rho_mat

    def _neg_t_copula_loglik(
        self,
        df_val: float,
        U_arr: np.ndarray,
        R: np.ndarray,
        R_inv: np.ndarray,
        logdet_R: float,
    ) -> float:
        """
        Negative log-likelihood of t-copula for given df.

        Copula log-likelihood:
            L = sum_t [ log f_{nu,R}(x_t) - sum_i log f_nu(x_ti) ]
        where x_it = t_nu^{-1}(u_it).

        Parameters
        ----------
        df_val : float
            Degrees of freedom candidate.
        U_arr : np.ndarray
            Uniform margin data, shape (T, N).
        R : np.ndarray
            Correlation matrix (N, N) — not used directly, only R_inv / logdet.
        R_inv : np.ndarray
            Pre-computed R^{-1}.
        logdet_R : float
            Pre-computed log|R|.

        Returns
        -------
        float
            Negative copula log-likelihood (to be minimised).
        """
        df = float(df_val)
        T, N = U_arr.shape
        eps = 1e-10
        X = t_dist.ppf(np.clip(U_arr, eps, 1.0 - eps), df=df)

        # Multivariate t log-density constant
        mv_const = (
            gammaln((df + N) / 2.0)
            - gammaln(df / 2.0)
            - (N / 2.0) * np.log(df * np.pi)
            - 0.5 * logdet_R
        )
        # Quadratic forms  x_t' R^{-1} x_t
        quad = np.einsum("ti,ij,tj->t", X, R_inv, X)
        mv_ll = T * mv_const - ((df + N) / 2.0) * np.sum(np.log(1.0 + quad / df))

        # Univariate t log-density sum (marginal adjustment)
        univ_const = (
            gammaln((df + 1.0) / 2.0)
            - gammaln(df / 2.0)
            - 0.5 * np.log(df * np.pi)
        )
        univ_ll = T * N * univ_const - ((df + 1.0) / 2.0) * np.sum(np.log(1.0 + X**2 / df))

        return -(mv_ll - univ_ll)  # negate to minimise

    def _tail_dependence_pair(self, df: float, rho: float) -> float:
        """
        Compute the tail dependence coefficient for a single pair.

        Formula (t-copula, closed form):
            lambda = 2 * t_{df+1}( -sqrt( (df+1)(1-rho)/(1+rho) ) )

        Parameters
        ----------
        df : float
            Degrees of freedom.
        rho : float
            Pairwise correlation.

        Returns
        -------
        float
            Tail dependence coefficient in [0, 1].
        """
        if rho >= 1.0 - 1e-10:
            return 1.0
        if rho <= -1.0 + 1e-10:
            return 0.0
        arg = -np.sqrt((df + 1.0) * (1.0 - rho) / (1.0 + rho))
        return float(2.0 * t_dist.cdf(arg, df=df + 1.0))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        sector_returns: pd.DataFrame,
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> CopulaResult:
        """
        Fit a Student-t (or Gaussian) copula to sector return data.

        Algorithm
        ---------
        1. Transform returns to uniform margins via config.marginal_transform.
        2. Estimate (or accept) the copula correlation matrix.
        3. Estimate df via MLE if copula_type == "t" and config.estimate_df.
        4. Compute tail-dependence coefficients (closed form for t-copula).
        5. Simulate n_simulation_paths draws from the fitted copula.
        6. Back-transform uniform draws to returns via empirical quantile function.

        Parameters
        ----------
        sector_returns : pd.DataFrame
            Daily sector returns, shape (T, N). Columns are sector names.
        correlation_matrix : pd.DataFrame, optional
            If provided (e.g. from DCC-GARCH current_correlation), use
            directly instead of estimating from data.

        Returns
        -------
        CopulaResult

        Raises
        ------
        ValueError
            If copula_type is not "t" or "gaussian".
            If marginal_transform is not "empirical" or "normal".
        """
        config = self._config
        t_start = time.time()

        valid_types = {"t", "gaussian"}
        if config.copula_type not in valid_types:
            raise ValueError(
                f"copula_type must be one of {sorted(valid_types)}, "
                f"got '{config.copula_type}'."
            )
        valid_transforms = {"empirical", "normal"}
        if config.marginal_transform not in valid_transforms:
            raise ValueError(
                f"marginal_transform must be one of {sorted(valid_transforms)}, "
                f"got '{config.marginal_transform}'."
            )

        sector_names = list(sector_returns.columns)
        N = len(sector_names)
        T = len(sector_returns)
        returns_arr = sector_returns.values.astype(float)

        # ── Step 1: Marginal transformation ─────────────────────────────
        U_arr = self._to_uniform_margins(returns_arr, T)
        logger.debug(f"Uniform margins: shape={U_arr.shape}, range=[{U_arr.min():.4f}, {U_arr.max():.4f}]")

        # ── Step 2: Correlation matrix ───────────────────────────────────
        if correlation_matrix is not None:
            try:
                R = correlation_matrix.loc[sector_names, sector_names].values.astype(float)
            except KeyError as exc:
                logger.warning(
                    f"Provided correlation_matrix missing sectors ({exc}); "
                    "estimating from data instead."
                )
                R = self._estimate_correlation(U_arr)
        else:
            R = self._estimate_correlation(U_arr)

        corr_df = pd.DataFrame(R, index=sector_names, columns=sector_names)

        # Pre-compute for log-likelihood
        try:
            sign, logdet_R = np.linalg.slogdet(R)
            R_inv = np.linalg.inv(R)
            if sign <= 0:
                raise np.linalg.LinAlgError("Correlation matrix is not positive definite.")
        except np.linalg.LinAlgError as exc:
            logger.warning(f"Correlation matrix inversion issue ({exc}); applying nearest-PD.")
            R = self._nearest_positive_definite(R)
            _, logdet_R = np.linalg.slogdet(R)
            R_inv = np.linalg.inv(R)
            corr_df = pd.DataFrame(R, index=sector_names, columns=sector_names)

        # ── Step 3: Estimate degrees of freedom ──────────────────────────
        if config.copula_type == "t":
            if config.estimate_df:
                logger.info("Estimating t-copula degrees of freedom via MLE…")
                t_df = time.time()
                opt = minimize_scalar(
                    self._neg_t_copula_loglik,
                    bounds=config.df_search_bounds,
                    method="bounded",
                    args=(U_arr, R, R_inv, float(logdet_R)),
                    options={"xatol": 1e-4},
                )
                df = float(opt.x)
                logger.info(
                    f"df MLE: {df:.3f} in {time.time() - t_df:.2f}s "
                    f"(bounds={config.df_search_bounds})"
                )
            else:
                df = float(config.degrees_of_freedom)
        else:
            df = float("inf")  # Gaussian: infinite df, no tail dependence

        # ── Step 4: Tail dependence coefficients ─────────────────────────
        tail_dep = np.zeros((N, N))
        if config.copula_type == "t" and not np.isinf(df):
            for i in range(N):
                for j in range(N):
                    if i == j:
                        tail_dep[i, j] = 1.0
                    else:
                        tail_dep[i, j] = self._tail_dependence_pair(df, float(R[i, j]))

        tail_dep_upper = pd.DataFrame(tail_dep, index=sector_names, columns=sector_names)
        tail_dep_lower = tail_dep_upper.copy()  # symmetric for elliptical copulas

        # ── Step 5: Simulate from copula ────────────────────────────────
        rng = np.random.default_rng(config.random_seed)
        n = config.n_simulation_paths

        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            R_reg = R + 1e-6 * np.eye(N)
            L = np.linalg.cholesky(R_reg)

        Z = rng.standard_normal((n, N)) @ L.T  # MVN draws

        if config.copula_type == "t" and not np.isinf(df):
            S = rng.chisquare(df=df, size=n) / df  # chi2 scaling factor
            X = Z / np.sqrt(S[:, np.newaxis])       # MVT draws
            U_sim = t_dist.cdf(X, df=df)
        else:  # Gaussian copula
            U_sim = norm.cdf(Z)

        U_sim = np.clip(U_sim, 1e-10, 1.0 - 1e-10)

        # ── Step 6: Back-transform to return space ───────────────────────
        sim_ret = np.empty((n, N), dtype=float)
        for j in range(N):
            sim_ret[:, j] = np.quantile(
                returns_arr[:, j],
                U_sim[:, j],
                method="linear",
            )

        simulated_returns_df = pd.DataFrame(sim_ret, columns=sector_names)

        # ── Pairwise joint exceedance probabilities ──────────────────────
        below = U_sim < config.tail_quantile  # (n, N) boolean
        joint_exc = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                joint_exc[i, j] = float(np.mean(below[:, i] & below[:, j]))

        joint_exc_df = pd.DataFrame(joint_exc, index=sector_names, columns=sector_names)

        logger.info(
            f"StudentTCopula.fit() done in {time.time() - t_start:.2f}s — "
            f"type={config.copula_type}, df={df:.3f}, N={N}, T={T}, n_sim={n}"
        )

        return CopulaResult(
            copula_type=config.copula_type,
            correlation_matrix=corr_df,
            degrees_of_freedom=df,
            upper_tail_dependence=tail_dep_upper,
            lower_tail_dependence=tail_dep_lower,
            simulated_uniforms=U_sim,
            simulated_returns=simulated_returns_df,
            joint_exceedance_probs=joint_exc_df,
            config=config,
        )

    def simulate_conditional(
        self,
        result: CopulaResult,
        shocked_sector: str,
        shock_quantile: float,
    ) -> pd.DataFrame:
        """
        Conditional simulation: given shocked_sector is at shock_quantile,
        return the joint distribution of all sectors via rejection sampling.

        Keeps only paths where the shocked sector's uniform draw is below
        ``shock_quantile + epsilon``, where:
            epsilon = max(0.02, 5 / n_simulation_paths)

        Parameters
        ----------
        result : CopulaResult
            Output of fit().
        shocked_sector : str
            Name of the sector to condition on.
        shock_quantile : float
            CDF quantile representing the shock (e.g. 0.05 for 5th percentile).

        Returns
        -------
        pd.DataFrame
            Shape (n_accepted_paths, N). Conditional return draws for all sectors.
            Columns are sector names.

        Raises
        ------
        ValueError
            If shocked_sector is not in the fitted sectors.
            If fewer than 30 paths pass the conditioning filter.
        """
        sector_names = list(result.simulated_returns.columns)
        if shocked_sector not in sector_names:
            raise ValueError(
                f"shocked_sector '{shocked_sector}' not found in sectors: {sector_names}."
            )

        j = sector_names.index(shocked_sector)
        uniforms = result.simulated_uniforms         # (n_paths, N)
        n = result.config.n_simulation_paths
        epsilon = max(0.02, 5.0 / n)

        mask = uniforms[:, j] < shock_quantile + epsilon
        n_accepted = int(mask.sum())

        if n_accepted < 30:
            raise ValueError(
                f"Only {n_accepted} paths pass the conditioning filter for "
                f"'{shocked_sector}' at quantile={shock_quantile:.3f} "
                f"(threshold={shock_quantile + epsilon:.4f}). "
                "Increase n_simulation_paths or relax shock_quantile."
            )

        logger.debug(
            f"simulate_conditional: {n_accepted}/{n} paths accepted "
            f"for '{shocked_sector}' at quantile={shock_quantile:.3f}."
        )

        accepted = result.simulated_returns.values[mask]
        return pd.DataFrame(accepted, columns=sector_names)

    def get_joint_exceedance_probability(
        self,
        result: CopulaResult,
        sectors: list[str],
        quantile: float,
    ) -> float:
        """
        Estimate P(all listed sectors simultaneously below quantile) from
        Monte Carlo simulated returns.

        Parameters
        ----------
        result : CopulaResult
            Output of fit().
        sectors : list[str]
            Subset of sector names to check jointly.
        quantile : float
            Return quantile threshold (e.g. 0.05 for 5th percentile crash).

        Returns
        -------
        float
            Estimated joint exceedance probability in [0, 1].

        Raises
        ------
        ValueError
            If any sector in ``sectors`` is not in the fitted result.
        """
        if not sectors:
            return 0.0

        sim = result.simulated_returns
        missing = [s for s in sectors if s not in sim.columns]
        if missing:
            raise ValueError(
                f"Sectors not found in simulated returns: {missing}."
            )

        # Build joint condition: all sectors below their individual quantile thresholds
        joint = pd.Series(True, index=sim.index)
        for sector in sectors:
            threshold = float(sim[sector].quantile(quantile))
            joint = joint & (sim[sector] < threshold)

        return float(joint.mean())


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def _smoke_test() -> None:
        import numpy as np
        import pandas as pd

        np.random.seed(7)

        n_days, n_sectors = 500, 3
        dates = pd.date_range("2022-01-01", periods=n_days, freq="B")

        # Correlated returns with fat tails (simulated from MVT)
        rng = np.random.default_rng(7)
        R_true = np.array([
            [1.00, 0.65, 0.30],
            [0.65, 1.00, 0.25],
            [0.30, 0.25, 1.00],
        ])
        L_true = np.linalg.cholesky(R_true)
        Z = rng.standard_normal((n_days, n_sectors)) @ L_true.T
        chi2_draws = rng.chisquare(df=5.0, size=n_days) / 5.0
        X_mvt = Z / np.sqrt(chi2_draws[:, None])
        raw = X_mvt * 0.015  # scale to realistic daily returns

        sector_returns = pd.DataFrame(
            raw, index=dates,
            columns=["Technology", "Financials", "Energy"]
        )

        # ── t-copula, estimate_df=True (full path) ──────────────────────
        config = CopulaConfig(
            copula_type="t",
            estimate_df=True,
            n_simulation_paths=5000,
            random_seed=42,
            tail_quantile=0.05,
            marginal_transform="empirical",
            correlation_estimator="kendall",
        )
        copula = StudentTCopula(config)
        t0 = time.time()
        result = copula.fit(sector_returns)
        elapsed = time.time() - t0
        print(f"  fit() (t, MLE) completed in {elapsed:.2f}s")

        # ── Shape checks ────────────────────────────────────────────────
        N, n = 3, 5000
        assert result.simulated_uniforms.shape == (n, N), (
            f"simulated_uniforms shape: {result.simulated_uniforms.shape}"
        )
        assert result.simulated_returns.shape == (n, N), (
            f"simulated_returns shape: {result.simulated_returns.shape}"
        )
        assert result.correlation_matrix.shape == (N, N), (
            f"correlation_matrix shape: {result.correlation_matrix.shape}"
        )
        assert result.upper_tail_dependence.shape == (N, N)
        assert result.lower_tail_dependence.shape == (N, N)
        assert result.joint_exceedance_probs.shape == (N, N)
        print(f"  Shape checks passed ✓")

        # ── Degrees of freedom ──────────────────────────────────────────
        assert result.config.df_search_bounds[0] <= result.degrees_of_freedom <= result.config.df_search_bounds[1], (
            f"df={result.degrees_of_freedom} outside search bounds"
        )
        print(f"  Estimated df={result.degrees_of_freedom:.3f} ✓")

        # ── Correlation matrix invariants ────────────────────────────────
        R = result.correlation_matrix.values
        assert np.allclose(np.diag(R), 1.0, atol=1e-9), "Diagonal not 1"
        assert np.allclose(R, R.T, atol=1e-12), "Not symmetric"
        assert R.min() >= -1.0 - 1e-9 and R.max() <= 1.0 + 1e-9, "Out of [-1,1]"
        print("  Correlation matrix invariants ✓")

        # ── Simulated uniforms in (0, 1) ────────────────────────────────
        U = result.simulated_uniforms
        assert U.min() > 0.0 and U.max() < 1.0, f"Uniforms out of (0,1): [{U.min()},{U.max()}]"
        print("  Simulated uniforms in (0, 1) ✓")

        # ── Tail dependence in [0, 1], diagonal = 1 ────────────────────
        td = result.upper_tail_dependence.values
        assert np.allclose(np.diag(td), 1.0, atol=1e-9), "Tail dep diagonal not 1"
        assert td.min() >= 0.0 - 1e-9 and td.max() <= 1.0 + 1e-9, "Tail dep out of [0,1]"
        print(f"  Tail dependence range: [{td.min():.4f}, {td.max():.4f}] ✓")

        # ── Upper == Lower (symmetric for t-copula) ─────────────────────
        assert np.allclose(
            result.upper_tail_dependence.values,
            result.lower_tail_dependence.values,
            atol=1e-12,
        ), "Upper and lower tail dependence should be equal for t-copula"
        print("  Upper == lower tail dependence for t-copula ✓")

        # ── Joint exceedance probabilities ───────────────────────────────
        jep = result.joint_exceedance_probs.values
        assert jep.min() >= 0.0 and jep.max() <= 1.0, "Joint exceedance out of [0,1]"
        # Diagonal should be close to tail_quantile (P(X < q) ≈ q)
        diag_jep = np.diag(jep)
        assert np.allclose(diag_jep, config.tail_quantile, atol=0.01), (
            f"Diagonal joint exceedance ≈ tail_quantile expected: {diag_jep}"
        )
        print(f"  Joint exceedance probs: diag≈{config.tail_quantile} ✓")

        # ── simulate_conditional ────────────────────────────────────────
        cond_df = copula.simulate_conditional(result, "Technology", shock_quantile=0.05)
        assert isinstance(cond_df, pd.DataFrame)
        assert list(cond_df.columns) == ["Technology", "Financials", "Energy"]
        assert len(cond_df) >= 30, f"Too few conditional paths: {len(cond_df)}"
        print(f"  simulate_conditional: {len(cond_df)} paths accepted ✓")

        # Shocked sector's median return should be below zero
        assert cond_df["Technology"].median() < sector_returns["Technology"].median(), (
            "Conditioned median should be lower than unconditional"
        )
        print("  Conditional median is below unconditional ✓")

        # ── simulate_conditional with invalid sector raises ValueError ───
        try:
            copula.simulate_conditional(result, "Utilities", shock_quantile=0.05)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass
        print("  simulate_conditional: invalid sector raises ValueError ✓")

        # ── get_joint_exceedance_probability ────────────────────────────
        p_joint = copula.get_joint_exceedance_probability(
            result, ["Technology", "Financials"], quantile=0.05
        )
        assert 0.0 <= p_joint <= 1.0, f"Joint prob out of [0,1]: {p_joint}"
        p_single = copula.get_joint_exceedance_probability(
            result, ["Technology"], quantile=0.05
        )
        # Joint(2 sectors) should be <= single sector
        assert p_joint <= p_single + 1e-9, (
            f"P(2-sector joint)={p_joint} > P(single)={p_single}"
        )
        print(f"  Joint P(Tech∩Fin < 5th pct) = {p_joint:.4f} ✓")
        assert copula.get_joint_exceedance_probability(result, [], quantile=0.05) == 0.0
        print("  Empty sector list returns 0.0 ✓")

        # ── Nearest PD on a non-PD matrix ───────────────────────────────
        bad = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]])
        bad[0, 1] = bad[1, 0] = 0.99  # not PD
        bad[0, 2] = bad[2, 0] = 0.99
        bad[1, 2] = bad[2, 1] = 0.99
        npd = copula._nearest_positive_definite(bad)
        assert np.allclose(np.diag(npd), 1.0, atol=1e-9), "PD fix: diagonal not 1"
        eigvals = np.linalg.eigvalsh(npd)
        assert eigvals.min() > 0, f"PD fix: still not PD: min_eig={eigvals.min()}"
        print(f"  _nearest_positive_definite: min_eig={eigvals.min():.2e} ✓")

        # ── Gaussian copula path ────────────────────────────────────────
        config_gauss = CopulaConfig(
            copula_type="gaussian",
            n_simulation_paths=2000,
            random_seed=99,
        )
        result_gauss = StudentTCopula(config_gauss).fit(sector_returns)
        assert result_gauss.copula_type == "gaussian"
        assert np.isinf(result_gauss.degrees_of_freedom)
        assert result_gauss.upper_tail_dependence.values.max() == 0.0, (
            "Gaussian copula should have zero tail dependence (off-diagonal)"
        )
        print("  Gaussian copula: zero tail dependence off-diagonal ✓")

        # ── DCC correlation matrix override ─────────────────────────────
        dcc_corr = pd.DataFrame(
            np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.0]]),
            index=["Technology", "Financials", "Energy"],
            columns=["Technology", "Financials", "Energy"],
        )
        config_override = CopulaConfig(estimate_df=False, n_simulation_paths=1000)
        result_override = StudentTCopula(config_override).fit(
            sector_returns, correlation_matrix=dcc_corr
        )
        assert np.allclose(
            result_override.correlation_matrix.values, dcc_corr.values, atol=1e-9
        ), "Overridden correlation matrix was not used"
        print("  Correlation matrix override works ✓")

        print("\n✓ [StudentTCopula] smoke test passed")

    _smoke_test()
