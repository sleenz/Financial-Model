"""
Leontief contagion propagation engine for the IDX macro stress model.

Propagates initial sector distress h(0) through an inter-sector weight
matrix W until convergence or the iteration limit, with nonlinear
saturation and optional IDR feedback loop.
Provides spectral-radius-based cascade risk classification and
networkx-based systemic-importance metrics.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import solve as _scipy_solve
from sklearn.linear_model import LinearRegression

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Optional dependency guards ────────────────────────────────────────────────

try:
    import networkx as nx
    _NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    _NETWORKX_AVAILABLE = False
    logger.warning("networkx not installed — graph-theoretic metrics will be skipped")

try:
    import joblib as _joblib
    _JOBLIB_AVAILABLE = True
except ImportError:
    _joblib = None
    _JOBLIB_AVAILABLE = False
    logger.warning("joblib not installed — fitted model caching disabled")

try:
    import sqlite3 as _sqlite3
    _SQLITE3_AVAILABLE = True
except ImportError:
    _sqlite3 = None
    _SQLITE3_AVAILABLE = False

try:
    import fredapi as _fredapi
    _FREDAPI_AVAILABLE = True
except ImportError:
    _fredapi = None
    _FREDAPI_AVAILABLE = False


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ContagionConfig:
    """
    Configuration for LeontifContagionEngine.

    Parameters
    ----------
    normalization : str
        "spectral": W = beta / (spectral_radius + epsilon) where
        epsilon = safety_margin / (1 - safety_margin).
        "row_sum": W[i,:] = M[i,:] / max(1, sum(M[i,:])).
    spectral_safety_margin : float
        Target: spectral_radius(W) <= 1 - safety_margin. Default 0.05.
    max_iterations : int
        Maximum Leontief iterations. Default 100.
    convergence_tol : float
        Stop when ||h(t+1) - h(t)||_2 < tol. Default 1e-6.
    distress_floor : float
        Clip h_i below this after each iteration. Default 0.0.
    distress_ceiling : float
        Clip h_i above this after each iteration. Default 1.0.
    cascade_warning_threshold : float
        spectral_radius > this → "warning". Default 0.90.
    cascade_critical_threshold : float
        spectral_radius > this → "critical". Default 0.98.
    idr_feedback_enabled : bool
        Enable equity → IDR → equity feedback loop. Default True.
    idr_sensitivity_window_days : int
        Window for estimating IDR-equity relationship. Default 504 (2Y).
    idr_feedback_lag_days : int
        Lag in days between equity move and IDR response. Default 5.
    compute_leontief_inverse : bool
        Compute (I-W)^{-1} for closed-form total impact. Default True.
    cache_dir : str
        Directory for joblib serialisation of fitted W and idr_params.
    """

    normalization: str = field(default="spectral")
    spectral_safety_margin: float = field(default=0.05)
    max_iterations: int = field(default=100)
    convergence_tol: float = field(default=1e-6)
    distress_floor: float = field(default=0.0)
    distress_ceiling: float = field(default=1.0)
    cascade_warning_threshold: float = field(default=0.90)
    cascade_critical_threshold: float = field(default=0.98)
    idr_feedback_enabled: bool = field(default=True)
    idr_sensitivity_window_days: int = field(default=504)
    idr_feedback_lag_days: int = field(default=5)
    compute_leontief_inverse: bool = field(default=True)
    cache_dir: str = field(default="data/model_cache")


# ── IDR feedback ──────────────────────────────────────────────────────────────

@dataclass
class IDRFeedbackParams:
    """
    Parameters for the equity → IDR → equity feedback loop.

    Attributes
    ----------
    equity_to_idr_coef : float
        Regression coefficient: how much JCI selloff weakens IDR.
        Positive = equity market down → IDR weaker (higher IDR/USD).
    idr_to_sector : pd.Series
        IDR sensitivity per sector (S matrix column for IDR_USD).
        Index = sector names.
    r_squared : float
        Quality of the equity → IDR regression.
    n_observations : int
        Number of data points used for regression.
    estimation_date : str
        ISO timestamp.
    """

    equity_to_idr_coef: float
    idr_to_sector: pd.Series
    r_squared: float
    n_observations: int
    estimation_date: str


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class ContagionResult:
    """
    Full output of LeontifContagionEngine.propagate().

    Attributes
    ----------
    h_initial : pd.Series
        First-order distress from macro sensitivity S · shock.
        Positive = sector gains; negative = sector loses from this shock.
    h_final : pd.Series
        Final state after nonlinear iterative propagation,
        clipped to [distress_floor, distress_ceiling].
    h_path : pd.DataFrame
        (n_iterations, N_sectors) distress path.
    converged : bool
        Whether the iteration converged before max_iterations.
    n_iterations : int
        Number of iterations performed.
    cascade_risk : str
        "low" | "warning" | "critical" based on spectral_radius of W.
    spectral_radius : float
        Spectral radius of the normalized weight matrix W.
    leontief_total : pd.Series
        Total impact via (I-W)^{-1} · h_initial (signed, both gains and losses).
        Valid only when spectral_radius(W) < 1.
    multiplier_table : pd.DataFrame
        Columns: Direct, Total, Multiplier — one row per sector.
    systemic_importance : pd.Series
        Eigenvector centrality per sector (requires networkx).
    contagion_paths : pd.DataFrame
        Betweenness centrality per sector (requires networkx).
    idr_feedback_rounds : int
        How many iterations had IDR feedback applied.
    warnings : list[str]
    config : ContagionConfig
    """

    h_initial: pd.Series
    h_final: pd.Series
    h_path: pd.DataFrame
    converged: bool
    n_iterations: int
    cascade_risk: str
    spectral_radius: float
    leontief_total: pd.Series
    multiplier_table: pd.DataFrame
    systemic_importance: pd.Series
    contagion_paths: pd.DataFrame
    idr_feedback_rounds: int
    warnings: list[str]
    config: ContagionConfig


# ── Engine ────────────────────────────────────────────────────────────────────

class LeontifContagionEngine:
    """
    Leontief contagion propagation engine.

    Build the inter-sector weight matrix W from a beta matrix,
    then propagate initial distress through the network using the
    nonlinear Leontief update rule with optional IDR feedback.

    Parameters
    ----------
    config : ContagionConfig
        Runtime configuration.
    """

    def __init__(self, config: ContagionConfig = ContagionConfig()) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def build_weight_matrix(
        self,
        beta_matrix: pd.DataFrame,
    ) -> tuple[pd.DataFrame, float]:
        """
        Build and normalize the inter-sector weight matrix W.

        Algorithm
        ---------
        1. Take absolute values of beta matrix.
        2. Zero the diagonal.
        3. Compute spectral_radius = max(|eigenvalues|) of raw matrix.
        4. Normalize so spectral_radius(W) < 1 - safety_margin.
           "spectral": W = M / (spectral_radius + epsilon),
                       epsilon = safety_margin / (1 - safety_margin).
                       If this does not reach the target, fall back to
                       W = M / spectral_radius * (1 - safety_margin).
           "row_sum":  W[i,:] = M[i,:] / max(1, sum(M[i,:])).
        5. Re-compute and verify spectral_radius(W) < 1.

        Parameters
        ----------
        beta_matrix : pd.DataFrame
            (N, N) cross-sector beta matrix. Index = columns = sector names.

        Returns
        -------
        tuple[pd.DataFrame, float]
            (W_normalized, spectral_radius_after_normalization)

        Raises
        ------
        ValueError
            If normalization fails to bring spectral_radius < 1.
        """
        sectors = list(beta_matrix.index)
        M = np.abs(beta_matrix.values.astype(float))
        np.fill_diagonal(M, 0.0)

        # Spectral radius of raw M
        eigvals_raw = np.linalg.eigvals(M)
        sr_raw = float(np.max(np.abs(eigvals_raw)))
        logger.debug(f"build_weight_matrix: spectral_radius(M_raw)={sr_raw:.4f}")

        margin = self._config.spectral_safety_margin
        target = 1.0 - margin

        if self._config.normalization == "spectral":
            if sr_raw < 1e-10:
                # Matrix is effectively zero — return as-is
                W = M.copy()
            else:
                epsilon = margin / (1.0 - margin)
                W = M / (sr_raw + epsilon)
                # Verify target was reached
                sr_check = float(np.max(np.abs(np.linalg.eigvals(W))))
                if sr_check >= target:
                    # Fallback: direct spectral scaling to exactly (1 - margin)
                    logger.debug(
                        f"build_weight_matrix: spectral formula gave sr={sr_check:.4f} "
                        f"> target={target:.4f}; using direct scaling"
                    )
                    W = M / sr_raw * target

        elif self._config.normalization == "row_sum":
            row_sums = M.sum(axis=1, keepdims=True)
            W = M / np.maximum(1.0, row_sums)
        else:
            raise ValueError(
                f"Unknown normalization '{self._config.normalization}'. "
                "Use 'spectral' or 'row_sum'."
            )

        # Final verification
        eigvals_W = np.linalg.eigvals(W)
        sr_W = float(np.max(np.abs(eigvals_W)))
        if sr_W >= 1.0:
            raise ValueError(
                f"Normalization failed: spectral_radius(W) = {sr_W:.4f} >= 1.0. "
                "This prevents Leontief convergence. "
                "Try reducing beta magnitudes or using row_sum normalization."
            )

        logger.info(
            f"build_weight_matrix: spectral_radius(W)={sr_W:.4f} "
            f"(target<{target:.2f}), sectors={len(sectors)}"
        )
        W_df = pd.DataFrame(W, index=sectors, columns=sectors)
        return W_df, sr_W

    def estimate_idr_feedback(
        self,
        sector_returns: pd.DataFrame,
        idr_returns: pd.Series,
        idr_sensitivity: pd.Series,
    ) -> IDRFeedbackParams:
        """
        Estimate the equity → IDR → equity feedback coefficient.

        Algorithm
        ---------
        1. Compute JCI proxy as equal-weight mean of all sector returns.
        2. Resample to weekly if daily.
        3. Regress IDR weekly pct change on lagged JCI return.
        4. equity_to_idr_coef = beta from this regression.
        5. idr_to_sector = idr_sensitivity (from S matrix column IDR_USD).

        Parameters
        ----------
        sector_returns : pd.DataFrame
            Daily or weekly sector return series.
        idr_returns : pd.Series
            Weekly IDR/USD pct change (from MacroDataResult.aligned_weekly["IDR_USD"]).
        idr_sensitivity : pd.Series
            S matrix column for IDR_USD — how each sector responds to IDR change.

        Returns
        -------
        IDRFeedbackParams
        """
        estimation_date = datetime.now().isoformat()

        # Build JCI proxy
        if sector_returns.empty or idr_returns.empty:
            logger.warning(
                "estimate_idr_feedback: empty input — IDR feedback disabled"
            )
            return IDRFeedbackParams(
                equity_to_idr_coef=0.0,
                idr_to_sector=idr_sensitivity,
                r_squared=0.0,
                n_observations=0,
                estimation_date=estimation_date,
            )

        jci = sector_returns.mean(axis=1)
        if len(jci) > 260:
            jci_weekly = jci.resample("W").sum()
        else:
            jci_weekly = jci.copy()

        # Lag in weeks (config.idr_feedback_lag_days → weeks)
        lag_weeks = max(1, self._config.idr_feedback_lag_days // 7)
        jci_lagged = jci_weekly.shift(lag_weeks)

        # Align with IDR returns
        common_idx = jci_lagged.index.intersection(idr_returns.index)
        if len(common_idx) < 20:
            logger.warning(
                f"estimate_idr_feedback: only {len(common_idx)} aligned "
                "observations — using zero coefficient"
            )
            return IDRFeedbackParams(
                equity_to_idr_coef=0.0,
                idr_to_sector=idr_sensitivity,
                r_squared=0.0,
                n_observations=len(common_idx),
                estimation_date=estimation_date,
            )

        df = pd.DataFrame(
            {"idr": idr_returns.loc[common_idx], "jci_lag": jci_lagged.loc[common_idx]}
        ).dropna()

        if len(df) < 10:
            return IDRFeedbackParams(
                equity_to_idr_coef=0.0,
                idr_to_sector=idr_sensitivity,
                r_squared=0.0,
                n_observations=len(df),
                estimation_date=estimation_date,
            )

        X = df[["jci_lag"]].values
        y = df["idr"].values
        try:
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            coef = float(model.coef_[0])
            r2 = float(model.score(X, y))
        except Exception as exc:
            logger.warning(f"estimate_idr_feedback regression failed: {exc}")
            coef = 0.0
            r2 = 0.0

        logger.info(
            f"estimate_idr_feedback: equity_to_idr_coef={coef:.4f}, "
            f"R2={r2:.3f}, n={len(df)}"
        )
        return IDRFeedbackParams(
            equity_to_idr_coef=coef,
            idr_to_sector=idr_sensitivity,
            r_squared=r2,
            n_observations=len(df),
            estimation_date=estimation_date,
        )

    def propagate(
        self,
        h_initial: pd.Series,
        W: pd.DataFrame,
        idr_params: Optional[IDRFeedbackParams] = None,
        weights: Optional[pd.Series] = None,
    ) -> ContagionResult:
        """
        Run Leontief contagion propagation.

        Algorithm
        ---------
        Main loop (t = 0 to max_iterations):
            1. h(t+1) = h(0) + (W · h(t)) * (1 - h(t))
               Nonlinear saturation: a sector at 90% distress absorbs
               at most 10% more incoming contagion.
            2. If idr_feedback_enabled and idr_params provided:
               delta_idr = idr_params.equity_to_idr_coef
                           * dot(h(t), sector_weights)
               h(t+1) += idr_params.idr_to_sector * delta_idr
            3. Clip h(t+1) to [distress_floor, distress_ceiling].
            4. Convergence: ||h(t+1) - h(t)||_2 < convergence_tol → break.

        Post-propagation:
        - If compute_leontief_inverse: solve (I-W)x = h_initial for x.
        - Build multiplier_table.
        - Compute networkx centrality metrics (if networkx available).

        Parameters
        ----------
        h_initial : pd.Series
            Index = sector names. Values = initial impact (signed decimal return).
            From MacroSensitivityEstimator.get_initial_distress().
        W : pd.DataFrame
            Normalized weight matrix from build_weight_matrix().
        idr_params : IDRFeedbackParams, optional
            If None and idr_feedback_enabled: log warning and skip loop.
        weights : pd.Series, optional
            Per-sector portfolio weights for IDR feedback. Equal weights if None.

        Returns
        -------
        ContagionResult
        """
        t0 = time.time()
        warnings_out: list[str] = []

        sectors = list(h_initial.index)
        n = len(sectors)
        W_arr = W.reindex(index=sectors, columns=sectors).fillna(0.0).values
        h_0 = h_initial.reindex(sectors).fillna(0.0).values.copy()

        # Sector weights for IDR feedback
        if weights is not None:
            w_arr = weights.reindex(sectors).fillna(0.0).values
            total_w = w_arr.sum()
            w_arr = w_arr / total_w if total_w > 0 else np.ones(n) / n
        else:
            w_arr = np.ones(n) / n

        # IDR sensitivity aligned to sectors
        idr_to_sector_arr = np.zeros(n)
        if (
            self._config.idr_feedback_enabled
            and idr_params is not None
            and not idr_params.idr_to_sector.empty
        ):
            idr_aligned = idr_params.idr_to_sector.reindex(sectors).fillna(0.0)
            idr_to_sector_arr = idr_aligned.values
        elif self._config.idr_feedback_enabled and idr_params is None:
            warnings_out.append(
                "IDR feedback enabled but idr_params is None — feedback loop skipped"
            )

        # Spectral radius of W
        try:
            eigvals_W = np.linalg.eigvals(W_arr)
            spectral_radius = float(np.max(np.abs(eigvals_W)))
        except Exception:
            spectral_radius = 0.0

        cascade_risk = self.get_cascade_risk_label(spectral_radius)
        if cascade_risk != "low":
            msg = (
                f"Cascade risk: {cascade_risk.upper()} "
                f"(spectral_radius={spectral_radius:.4f})"
            )
            warnings_out.append(msg)
            logger.warning(msg)

        # ── Iterative propagation ─────────────────────────────────────────────
        h_t = h_0.copy()
        h_path_list: list[np.ndarray] = []
        converged = False
        n_iter = 0
        idr_feedback_rounds = 0

        floor = self._config.distress_floor
        ceiling = self._config.distress_ceiling
        tol = self._config.convergence_tol

        for t in range(self._config.max_iterations):
            h_path_list.append(h_t.copy())

            # Core Leontief update with saturation
            contagion = W_arr @ h_t
            saturation = 1.0 - h_t
            h_next = h_0 + contagion * saturation

            # IDR feedback
            if (
                self._config.idr_feedback_enabled
                and idr_params is not None
                and np.any(idr_to_sector_arr != 0)
            ):
                portfolio_distress = float(np.dot(h_t, w_arr))
                delta_idr = idr_params.equity_to_idr_coef * portfolio_distress
                h_next = h_next + idr_to_sector_arr * delta_idr
                idr_feedback_rounds += 1

            # Clip
            h_next = np.clip(h_next, floor, ceiling)

            # Convergence
            delta_norm = float(np.linalg.norm(h_next - h_t))
            h_t = h_next
            n_iter = t + 1

            if delta_norm < tol:
                converged = True
                break

        h_final = pd.Series(h_t, index=sectors)
        h_path_arr = np.vstack(h_path_list) if h_path_list else np.zeros((1, n))
        h_path_df = pd.DataFrame(h_path_arr, columns=sectors)

        # ── Leontief inverse (closed-form) ────────────────────────────────────
        leontief_total = h_initial.copy()
        if self._config.compute_leontief_inverse and spectral_radius < 1.0:
            try:
                I_minus_W = np.eye(n) - W_arr
                lt_arr = _scipy_solve(I_minus_W, h_0)
                leontief_total = pd.Series(lt_arr, index=sectors)
            except Exception as exc:
                warnings_out.append(
                    f"Leontief inverse failed: {exc} — using h_initial as total"
                )
                logger.warning(f"propagate: Leontief inverse failed: {exc}")

        # ── Multiplier table ──────────────────────────────────────────────────
        with np.errstate(divide="ignore", invalid="ignore"):
            direct_vals = h_initial.reindex(sectors).fillna(0.0)
            total_vals = leontief_total.reindex(sectors).fillna(0.0)
            ratio = np.where(
                np.abs(direct_vals.values) > 1e-10,
                total_vals.values / direct_vals.values,
                1.0,
            )
        multiplier_table = pd.DataFrame(
            {
                "Direct": direct_vals.values,
                "Total": total_vals.values,
                "Multiplier": ratio,
            },
            index=sectors,
        )

        # ── NetworkX metrics ──────────────────────────────────────────────────
        systemic_importance = pd.Series(1.0 / max(n, 1), index=sectors)
        contagion_paths = pd.DataFrame(
            {"betweenness": np.zeros(n)}, index=sectors
        )
        if _NETWORKX_AVAILABLE:
            try:
                G = self.build_networkx_graph(W)
                ec = nx.eigenvector_centrality_numpy(G, weight="weight")
                systemic_importance = pd.Series(ec).reindex(sectors).fillna(0.0)
                bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
                contagion_paths = pd.DataFrame(
                    {"betweenness": [bc.get(s, 0.0) for s in sectors]},
                    index=sectors,
                )
            except Exception as exc:
                warnings_out.append(f"NetworkX metrics failed: {exc}")
                logger.debug(f"propagate: NetworkX metrics failed: {exc}")

        elapsed = time.time() - t0
        logger.info(
            f"propagate(): converged={converged}, n_iter={n_iter}, "
            f"spectral_radius={spectral_radius:.4f}, cascade={cascade_risk}, "
            f"elapsed={elapsed:.3f}s"
        )

        return ContagionResult(
            h_initial=h_initial,
            h_final=h_final,
            h_path=h_path_df,
            converged=converged,
            n_iterations=n_iter,
            cascade_risk=cascade_risk,
            spectral_radius=spectral_radius,
            leontief_total=leontief_total,
            multiplier_table=multiplier_table,
            systemic_importance=systemic_importance,
            contagion_paths=contagion_paths,
            idr_feedback_rounds=idr_feedback_rounds,
            warnings=warnings_out,
            config=self._config,
        )

    def get_cascade_risk_label(self, spectral_radius: float) -> str:
        """
        Return "low" | "warning" | "critical" based on spectral_radius thresholds.

        Parameters
        ----------
        spectral_radius : float
            Spectral radius of the normalized weight matrix W.

        Returns
        -------
        str
        """
        if spectral_radius <= self._config.cascade_warning_threshold:
            return "low"
        if spectral_radius <= self._config.cascade_critical_threshold:
            return "warning"
        return "critical"

    def build_networkx_graph(self, W: pd.DataFrame) -> "nx.DiGraph":
        """
        Build a directed weighted networkx graph from W.

        Node attributes: sector name (label).
        Edge attributes: weight = W[source][target].
        Computes and attaches eigenvector_centrality and betweenness_centrality
        as node attributes on the returned graph.

        Parameters
        ----------
        W : pd.DataFrame
            (N, N) normalized weight matrix.

        Returns
        -------
        nx.DiGraph
            Directed graph. Returns an empty DiGraph if networkx is unavailable.
        """
        if not _NETWORKX_AVAILABLE:
            logger.warning("networkx unavailable — returning empty graph")
            import types
            stub = types.SimpleNamespace()
            stub.nodes = {}
            stub.edges = {}
            return stub

        G = nx.DiGraph()
        sectors = list(W.index)
        G.add_nodes_from(sectors)
        for i, src in enumerate(sectors):
            for j, dst in enumerate(sectors):
                w = float(W.iloc[i, j])
                if w > 0 and src != dst:
                    G.add_edge(src, dst, weight=w)

        # Attach centrality as node attributes
        try:
            ec = nx.eigenvector_centrality_numpy(G, weight="weight")
            nx.set_node_attributes(G, ec, "eigenvector_centrality")
        except Exception:
            nx.set_node_attributes(G, {s: 0.0 for s in sectors}, "eigenvector_centrality")

        try:
            bc = nx.betweenness_centrality(G, weight="weight", normalized=True)
            nx.set_node_attributes(G, bc, "betweenness_centrality")
        except Exception:
            nx.set_node_attributes(G, {s: 0.0 for s in sectors}, "betweenness_centrality")

        return G

    def save_fitted(
        self, W: pd.DataFrame, idr_params: IDRFeedbackParams
    ) -> None:
        """
        Serialize W and idr_params to config.cache_dir via joblib.

        Parameters
        ----------
        W : pd.DataFrame
            Normalized weight matrix.
        idr_params : IDRFeedbackParams
        """
        if not _JOBLIB_AVAILABLE:
            logger.warning("joblib not available — fitted model not saved")
            return
        try:
            os.makedirs(self._config.cache_dir, exist_ok=True)
            path = os.path.join(self._config.cache_dir, "contagion_fitted.joblib")
            payload = {
                "W": W,
                "idr_params": idr_params,
                "timestamp": time.time(),
            }
            _joblib.dump(payload, path)
            logger.info(f"ContagionEngine: saved fitted model to {path}")
        except Exception as exc:
            logger.error(f"save_fitted() failed: {exc}")

    def load_fitted(
        self,
    ) -> Optional[tuple[pd.DataFrame, IDRFeedbackParams]]:
        """
        Load serialized W and idr_params from config.cache_dir.

        Returns
        -------
        tuple[pd.DataFrame, IDRFeedbackParams] or None
            None if the cache file does not exist or loading fails.
        """
        if not _JOBLIB_AVAILABLE:
            return None
        path = os.path.join(self._config.cache_dir, "contagion_fitted.joblib")
        if not os.path.exists(path):
            return None
        try:
            data = _joblib.load(path)
            return data["W"], data["idr_params"]
        except Exception as exc:
            logger.warning(f"load_fitted() failed: {exc}")
            return None


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    def _smoke_test() -> None:
        rng = np.random.default_rng(0)

        sectors = ["Technology", "Financials", "Energy", "BasicMaterials"]
        n = len(sectors)

        # Synthetic beta matrix (diagonal = 1, off-diagonal small)
        beta = rng.uniform(0.2, 0.8, (n, n))
        np.fill_diagonal(beta, 1.0)
        beta_df = pd.DataFrame(beta, index=sectors, columns=sectors)

        engine = LeontifContagionEngine()

        # build_weight_matrix
        W, sr = engine.build_weight_matrix(beta_df)
        assert sr < 1.0, f"spectral_radius must be < 1.0, got {sr:.4f}"
        assert W.shape == (n, n), f"W shape mismatch: {W.shape}"
        assert not W.isnull().any().any(), "W must not have NaN"
        print(f"  build_weight_matrix: spectral_radius={sr:.4f}")

        # Row-sum normalization
        engine_rs = LeontifContagionEngine(
            ContagionConfig(normalization="row_sum")
        )
        W_rs, sr_rs = engine_rs.build_weight_matrix(beta_df)
        assert sr_rs < 1.0, f"row_sum sr={sr_rs:.4f} must be < 1"
        print(f"  row_sum normalization: spectral_radius={sr_rs:.4f}")

        # estimate_idr_feedback
        T = 200
        dates = pd.date_range("2020-01-10", periods=T, freq="W")
        sector_ret = pd.DataFrame(rng.normal(0, 0.01, (T, n)), index=dates, columns=sectors)
        idr_series = pd.Series(rng.normal(0, 0.005, T), index=dates, name="IDR_USD")
        idr_sens = pd.Series(rng.uniform(0.1, 0.5, n), index=sectors)
        idr_params = engine.estimate_idr_feedback(sector_ret, idr_series, idr_sens)
        assert isinstance(idr_params.equity_to_idr_coef, float)
        print(f"  IDR feedback coef={idr_params.equity_to_idr_coef:.4f}, R2={idr_params.r_squared:.3f}")

        # propagate
        h_initial = pd.Series(
            {"Technology": -0.05, "Financials": 0.10, "Energy": 0.08, "BasicMaterials": 0.02}
        )
        result = engine.propagate(h_initial, W, idr_params=idr_params)
        assert result.converged or result.n_iterations == engine._config.max_iterations
        assert len(result.h_final) == n
        assert result.spectral_radius < 1.0
        assert result.cascade_risk in ("low", "warning", "critical")
        assert result.multiplier_table.shape == (n, 3)
        print(f"  propagate: converged={result.converged}, n_iter={result.n_iterations}")
        print(f"  cascade_risk={result.cascade_risk}, sr={result.spectral_radius:.4f}")
        print(f"  leontief_total:\n{result.leontief_total}")

        # Commodity Boom: Energy should gain (positive leontief_total for Energy)
        h_boom = pd.Series(
            {"Technology": 0.0, "Financials": 0.0, "Energy": 0.15, "BasicMaterials": 0.10}
        )
        result_boom = engine.propagate(h_boom, W)
        assert float(result_boom.leontief_total["Energy"]) > 0, (
            "Energy should have positive total in commodity boom"
        )
        print(f"  Commodity Boom: Energy leontief_total={result_boom.leontief_total['Energy']:.4f}")

        # Global Risk-Off: total_pnl < direct_pnl
        h_risk_off = pd.Series(
            {"Technology": -0.10, "Financials": -0.12, "Energy": -0.08, "BasicMaterials": -0.06}
        )
        test_weights = pd.Series(0.25, index=sectors)
        result_ro = engine.propagate(h_risk_off, W, weights=test_weights)
        direct_pnl = float((test_weights * result_ro.h_initial).sum())
        total_pnl = float((test_weights * result_ro.leontief_total).sum())
        assert total_pnl < direct_pnl, (
            f"Contagion should amplify losses: total={total_pnl:.4f} < direct={direct_pnl:.4f}"
        )
        print(f"  Risk-Off: direct_pnl={direct_pnl:.4f}, total_pnl={total_pnl:.4f}")

        # save/load
        engine.save_fitted(W, idr_params)
        loaded = engine.load_fitted()
        if loaded is not None:
            W_loaded, idr_loaded = loaded
            assert W_loaded.shape == W.shape
            print(f"  save/load: W shape={W_loaded.shape}")

        # get_cascade_risk_label
        assert engine.get_cascade_risk_label(0.5) == "low"
        assert engine.get_cascade_risk_label(0.92) == "warning"
        assert engine.get_cascade_risk_label(0.99) == "critical"
        print("  cascade risk labels: low/warning/critical OK")

        print("\ncontagion smoke test PASSED")

    _smoke_test()
