"""
Hidden Markov Model market regime detection.

Fits a Gaussian HMM to sector return features (rolling volatility, mean return,
vol-of-vol, cross-sector dispersion). States are relabelled post-fit by mean
conditional volatility so that state 0 is always "calm" and the highest index
is always "crisis". Regime-averaged DCC correlation matrices are extracted for
use in downstream stress testing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.risk.dcc_garch import DCCGARCHResult

try:
    from hmmlearn.hmm import GaussianHMM
    _HMMLEARN_AVAILABLE = True
except ImportError:
    GaussianHMM = None
    _HMMLEARN_AVAILABLE = False

logger = get_logger(__name__)

if not _HMMLEARN_AVAILABLE:
    logger.warning(
        "hmmlearn is not installed. Regime detection will be unavailable. "
        "Install with: pip install hmmlearn>=0.3.0"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class ConvergenceError(RuntimeError):
    """Raised when no HMM initialisation achieved convergence."""


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeConfig:
    """Configuration for Gaussian HMM regime detection."""

    n_states: int = field(default=3)
    covariance_type: str = field(default="full")  # "full"|"diag"|"tied"|"spherical"
    n_iter: int = field(default=200)
    tol: float = field(default=1e-4)
    n_init: int = field(default=10)
    random_seed: int = field(default=42)
    features: list = field(
        default_factory=lambda: ["rolling_vol", "mean_return"]
    )
    rolling_vol_window: int = field(default=21)
    regime_label_map: dict = field(
        default_factory=lambda: {
            2: {0: "calm", 1: "crisis"},
            3: {0: "calm", 1: "elevated", 2: "crisis"},
            4: {0: "calm", 1: "mild_stress", 2: "elevated", 3: "crisis"},
        }
    )
    # States are re-ordered after fitting so that the state with the lowest
    # mean conditional volatility is assigned index 0 ("calm"), and the state
    # with the highest mean conditional volatility is assigned the last index.


@dataclass
class RegimeResult:
    """Full output from MarketRegimeDetector.fit()."""

    state_sequence: pd.Series              # integer (relabelled) state per date
    state_probabilities: pd.DataFrame      # shape (T_valid, n_states)
    current_state: int
    current_state_label: str
    current_state_probability: float
    regime_statistics: pd.DataFrame        # index=state_label; cols=mean_vol, mean_return, n_days
    transition_matrix: np.ndarray          # (n_states, n_states)
    avg_regime_duration_days: dict         # {state_label: float}
    config: RegimeConfig
    fit_log_likelihood: float
    convergence_achieved: bool


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class MarketRegimeDetector:
    """
    Gaussian Hidden Markov Model for market regime detection.

    Fits a multi-state HMM to engineered sector return features.  States are
    relabelled by ascending mean conditional volatility: 0 → "calm",
    n_states-1 → "crisis".

    Parameters
    ----------
    config : RegimeConfig
        Runtime configuration for HMM structure, features, and relabelling.
    """

    def __init__(self, config: RegimeConfig = RegimeConfig()) -> None:
        self._config = config
        self._last_valid_dates: Optional[pd.Index] = None  # set by _build_features

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_features(self, sector_returns: pd.DataFrame) -> np.ndarray:
        """
        Build the feature matrix for HMM estimation.

        Supported features (controlled by config.features):

        * ``"rolling_vol"`` — rolling std of cross-sector mean return,
          window = config.rolling_vol_window. Standardised.
        * ``"mean_return"`` — cross-sector mean return per day. Standardised.
        * ``"vol_of_vol"`` — rolling std of rolling_vol. Standardised.
        * ``"cross_sector_dispersion"`` — cross-sector std per day. Standardised.

        NaN rows introduced by rolling operations are dropped from the front.
        The valid date index is stored as ``self._last_valid_dates``.

        Parameters
        ----------
        sector_returns : pd.DataFrame
            Daily sector returns, shape (T, N).

        Returns
        -------
        np.ndarray
            Shape (T_valid, n_features). Standardised feature matrix.

        Raises
        ------
        ValueError
            If config.features is empty or all features produce NaN-only series.
        """
        config = self._config
        mean_ret = sector_returns.mean(axis=1)

        series_list: list[pd.Series] = []
        for feat_name in config.features:
            if feat_name == "rolling_vol":
                s = mean_ret.rolling(config.rolling_vol_window).std()
            elif feat_name == "mean_return":
                s = mean_ret.copy()
            elif feat_name == "vol_of_vol":
                rv = mean_ret.rolling(config.rolling_vol_window).std()
                s = rv.rolling(config.rolling_vol_window).std()
            elif feat_name == "cross_sector_dispersion":
                s = sector_returns.std(axis=1)
            else:
                logger.warning(
                    f"_build_features: unknown feature '{feat_name}', skipping."
                )
                continue
            series_list.append(s.rename(feat_name))

        if not series_list:
            raise ValueError(
                "config.features produced no valid series. "
                f"Check feature names: {config.features}"
            )

        feature_df = pd.concat(series_list, axis=1).dropna()

        if feature_df.empty:
            raise ValueError(
                "Feature DataFrame is empty after dropping NaN rows. "
                "Increase the length of sector_returns or reduce rolling_vol_window."
            )

        # Standardise: zero mean, unit variance per feature
        col_means = feature_df.mean()
        col_stds = feature_df.std().replace(0.0, 1.0)
        feature_df = (feature_df - col_means) / col_stds

        self._last_valid_dates = feature_df.index
        return feature_df.values.astype(float)

    @staticmethod
    def _avg_run_length(states: np.ndarray, target: int) -> float:
        """Compute average consecutive-run length for integer state ``target``."""
        runs: list[int] = []
        count = 0
        for s in states:
            if int(s) == target:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                    count = 0
        if count > 0:
            runs.append(count)
        return float(np.mean(runs)) if runs else 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, sector_returns: pd.DataFrame) -> RegimeResult:
        """
        Fit Gaussian HMM to sector return features.

        Algorithm
        ---------
        1. Build feature matrix via ``_build_features``.
        2. Fit ``GaussianHMM`` config.n_init times with seeds
           config.random_seed … config.random_seed + n_init - 1.
        3. Keep the model with highest log-likelihood.
        4. Decode state sequence via Viterbi (``predict``).
        5. Compute smoothed state probabilities via forward-backward
           (``predict_proba``).
        6. Relabel states by ascending mean conditional volatility:
           lowest-vol state → index 0 ("calm").
        7. Compute per-regime statistics and average durations.
        8. Reorder transition matrix to match relabelled states.

        Parameters
        ----------
        sector_returns : pd.DataFrame
            Daily sector returns, shape (T, N).

        Returns
        -------
        RegimeResult

        Raises
        ------
        ImportError
            If hmmlearn is not installed.
        ConvergenceError
            If all ``n_init`` HMM initialisations fail to produce a result.
        """
        if not _HMMLEARN_AVAILABLE:
            raise ImportError(
                "hmmlearn is required for regime detection. "
                "Install with: pip install hmmlearn>=0.3.0"
            )

        config = self._config
        t_start = time.time()

        X = self._build_features(sector_returns)
        valid_dates: pd.Index = self._last_valid_dates  # set by _build_features
        T_valid, n_feat = X.shape
        logger.info(
            f"MarketRegimeDetector.fit(): T_valid={T_valid}, n_feat={n_feat}, "
            f"n_states={config.n_states}, n_init={config.n_init}"
        )

        # ── Step 2: n_init fits, keep best log-likelihood ────────────────
        best_model: Optional[GaussianHMM] = None
        best_ll: float = -np.inf
        convergence_achieved = False

        for init_idx in range(config.n_init):
            try:
                model = GaussianHMM(
                    n_components=config.n_states,
                    covariance_type=config.covariance_type,
                    n_iter=config.n_iter,
                    tol=config.tol,
                    random_state=config.random_seed + init_idx,
                )
                model.fit(X)
                ll = float(model.score(X))

                converged = bool(
                    getattr(getattr(model, "monitor_", None), "converged", True)
                )
                if ll > best_ll:
                    best_ll = ll
                    best_model = model
                if converged:
                    convergence_achieved = True

            except Exception as exc:
                logger.warning(
                    f"HMM init {init_idx + 1}/{config.n_init} failed: {exc}"
                )

        if best_model is None:
            raise ConvergenceError(
                f"All {config.n_init} HMM initialisations failed. "
                "Try reducing n_states, switching covariance_type to 'diag', "
                "or increasing the number of observations."
            )

        # ── Steps 4–5: Viterbi sequence + smoothed probs ────────────────
        raw_states: np.ndarray = best_model.predict(X)           # (T_valid,)
        raw_probs: np.ndarray = best_model.predict_proba(X)      # (T_valid, n_states)

        # ── Step 6: Relabel by mean conditional volatility ────────────────
        # Use "rolling_vol" feature if available, otherwise first feature
        if "rolling_vol" in config.features:
            vol_feat_idx = config.features.index("rolling_vol")
        else:
            vol_feat_idx = 0

        vol_feature = X[:, vol_feat_idx]
        state_vol_means = np.array([
            float(vol_feature[raw_states == s].mean())
            if (raw_states == s).any()
            else 0.0
            for s in range(config.n_states)
        ])
        # state_order[new_label] = original_label
        state_order: np.ndarray = np.argsort(state_vol_means)
        # old → new mapping
        state_map: dict[int, int] = {
            int(old): new for new, old in enumerate(state_order)
        }

        relabelled_states = np.array([state_map[int(s)] for s in raw_states])
        relabelled_probs = raw_probs[:, state_order]  # reorder columns

        # Regime label names
        label_map: dict[int, str] = config.regime_label_map.get(
            config.n_states,
            {i: f"state_{i}" for i in range(config.n_states)},
        )

        state_seq = pd.Series(relabelled_states, index=valid_dates, name="regime")
        state_probs_df = pd.DataFrame(
            relabelled_probs,
            index=valid_dates,
            columns=list(range(config.n_states)),
        )

        # Current (latest) state
        current_state_int = int(relabelled_states[-1])
        current_state_label = label_map.get(current_state_int, f"state_{current_state_int}")
        current_state_prob = float(relabelled_probs[-1, current_state_int])

        # ── Step 7: Regime statistics ────────────────────────────────────
        # Align mean return series to valid_dates
        mean_ret_full = sector_returns.mean(axis=1)
        mean_ret_valid = mean_ret_full.reindex(valid_dates).values

        stats_rows = []
        for state_int in range(config.n_states):
            mask = relabelled_states == state_int
            state_label = label_map.get(state_int, f"state_{state_int}")
            stats_rows.append(
                {
                    "mean_vol": float(vol_feature[mask].mean()) if mask.any() else 0.0,
                    "mean_return": float(mean_ret_valid[mask].mean()) if mask.any() else 0.0,
                    "n_days": int(mask.sum()),
                }
            )

        regime_stats_df = pd.DataFrame(
            stats_rows,
            index=[label_map.get(i, f"state_{i}") for i in range(config.n_states)],
        )

        # ── Average regime durations ─────────────────────────────────────
        avg_durations: dict[str, float] = {
            label_map.get(s, f"state_{s}"): self._avg_run_length(relabelled_states, s)
            for s in range(config.n_states)
        }

        # ── Reorder transition matrix ────────────────────────────────────
        old_transmat = best_model.transmat_
        new_transmat = old_transmat[state_order, :][:, state_order]

        logger.info(
            f"MarketRegimeDetector.fit() done in {time.time() - t_start:.2f}s — "
            f"current_regime='{current_state_label}' (p={current_state_prob:.2%}), "
            f"ll={best_ll:.2f}, converged={convergence_achieved}"
        )

        return RegimeResult(
            state_sequence=state_seq,
            state_probabilities=state_probs_df,
            current_state=current_state_int,
            current_state_label=current_state_label,
            current_state_probability=current_state_prob,
            regime_statistics=regime_stats_df,
            transition_matrix=new_transmat,
            avg_regime_duration_days=avg_durations,
            config=config,
            fit_log_likelihood=best_ll,
            convergence_achieved=convergence_achieved,
        )

    def get_regime_correlation(
        self,
        regime_label: str,
        dcc_result: "DCCGARCHResult",
        regime_result: RegimeResult,
    ) -> pd.DataFrame:
        """
        Extract the average DCC correlation matrix for all timesteps
        classified under ``regime_label``.

        Parameters
        ----------
        regime_label : str
            e.g. ``"calm"`` | ``"elevated"`` | ``"crisis"``.
        dcc_result : DCCGARCHResult
            Output of DCCGARCHModel.fit().
        regime_result : RegimeResult
            Output of MarketRegimeDetector.fit().

        Returns
        -------
        pd.DataFrame
            NxN average correlation matrix for that regime.
            Returns identity matrix (with a logged warning) if the regime has
            fewer than 5 aligned observations.

        Raises
        ------
        ValueError
            If ``regime_label`` is not in the config's regime_label_map.
        """
        config = regime_result.config
        label_map = config.regime_label_map.get(
            config.n_states,
            {i: f"state_{i}" for i in range(config.n_states)},
        )
        rev_label_map: dict[str, int] = {v: k for k, v in label_map.items()}

        if regime_label not in rev_label_map:
            raise ValueError(
                f"regime_label '{regime_label}' not found in label map "
                f"{label_map} for n_states={config.n_states}."
            )

        target_state_int = rev_label_map[regime_label]
        state_seq = regime_result.state_sequence  # pd.Series indexed by valid_dates
        mask = state_seq == target_state_int
        target_dates = state_seq.index[mask]

        # Align with DCC dates
        dcc_dates = dcc_result.conditional_volatilities.index
        common_dates = target_dates.intersection(dcc_dates)
        n_common = len(common_dates)

        sectors = dcc_result.sector_names
        N = len(sectors)

        if n_common < 5:
            logger.warning(
                f"get_regime_correlation: only {n_common} aligned observations "
                f"for regime '{regime_label}'; returning identity matrix."
            )
            return pd.DataFrame(np.eye(N), index=sectors, columns=sectors)

        # Map common dates to integer positions in dcc_result
        dcc_date_to_pos = {d: i for i, d in enumerate(dcc_dates)}
        dcc_positions = np.array([dcc_date_to_pos[d] for d in common_dates], dtype=int)

        avg_corr = dcc_result.conditional_correlations[dcc_positions].mean(axis=0)

        # Enforce diagonal = 1 (guard against floating-point drift)
        np.fill_diagonal(avg_corr, 1.0)

        logger.debug(
            f"get_regime_correlation('{regime_label}'): "
            f"{n_common} timesteps averaged."
        )
        return pd.DataFrame(avg_corr, index=sectors, columns=sectors)

    def get_current_regime_correlation(
        self,
        dcc_result: "DCCGARCHResult",
        regime_result: RegimeResult,
    ) -> pd.DataFrame:
        """
        Return the average DCC correlation matrix for the currently detected regime.

        Parameters
        ----------
        dcc_result : DCCGARCHResult
            Output of DCCGARCHModel.fit().
        regime_result : RegimeResult
            Output of MarketRegimeDetector.fit().

        Returns
        -------
        pd.DataFrame
            NxN correlation matrix for the current regime.
        """
        return self.get_regime_correlation(
            regime_label=regime_result.current_state_label,
            dcc_result=dcc_result,
            regime_result=regime_result,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    def _smoke_test() -> None:
        import numpy as np
        import pandas as pd

        np.random.seed(99)

        n_days, n_sectors = 600, 3
        dates = pd.date_range("2021-01-01", periods=n_days, freq="B")

        # Simulate 3 distinct volatility regimes
        regime_vols = [0.005, 0.012, 0.025]
        regime_lengths = [200, 200, 200]
        vol_series = np.concatenate(
            [np.full(l, v) for v, l in zip(regime_vols, regime_lengths)]
        )
        raw = np.random.randn(n_days, n_sectors) * vol_series[:, np.newaxis]
        sector_returns = pd.DataFrame(
            raw, index=dates,
            columns=["Technology", "Financials", "Energy"]
        )

        config = RegimeConfig(
            n_states=3,
            n_iter=100,
            n_init=5,
            random_seed=42,
            features=["rolling_vol", "mean_return"],
            rolling_vol_window=21,
        )
        detector = MarketRegimeDetector(config)

        # ── _build_features ──────────────────────────────────────────────
        X = detector._build_features(sector_returns)
        assert X.ndim == 2, f"Expected 2D array, got {X.ndim}D"
        assert X.shape[1] == 2, f"Expected 2 features, got {X.shape[1]}"
        assert not np.isnan(X).any(), "Feature matrix contains NaN"
        # Should lose rolling_vol_window-1 rows from front
        expected_rows = n_days - (config.rolling_vol_window - 1)
        assert X.shape[0] == expected_rows, (
            f"Expected {expected_rows} valid rows, got {X.shape[0]}"
        )
        print(f"  _build_features(): shape={X.shape}, no NaN ✓")

        # ── fit ──────────────────────────────────────────────────────────
        t0 = time.time()
        result = detector.fit(sector_returns)
        elapsed = time.time() - t0
        print(f"  fit() completed in {elapsed:.2f}s")

        # State sequence
        assert isinstance(result.state_sequence, pd.Series)
        assert len(result.state_sequence) == expected_rows, (
            f"state_sequence length mismatch: {len(result.state_sequence)}"
        )
        assert set(result.state_sequence.unique()).issubset({0, 1, 2}), (
            f"Unexpected state values: {result.state_sequence.unique()}"
        )
        print(f"  state_sequence: length={len(result.state_sequence)}, "
              f"unique={sorted(result.state_sequence.unique())} ✓")

        # State probabilities
        assert result.state_probabilities.shape == (expected_rows, 3), (
            f"state_probabilities shape: {result.state_probabilities.shape}"
        )
        prob_row_sums = result.state_probabilities.sum(axis=1)
        assert np.allclose(prob_row_sums, 1.0, atol=1e-6), (
            f"state_probabilities rows don't sum to 1: {prob_row_sums.describe()}"
        )
        print("  state_probabilities: shape correct, rows sum to 1 ✓")

        # Transition matrix
        assert result.transition_matrix.shape == (3, 3), (
            f"transition_matrix shape: {result.transition_matrix.shape}"
        )
        row_sums = result.transition_matrix.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6), (
            f"transition_matrix rows don't sum to 1: {row_sums}"
        )
        assert (result.transition_matrix >= 0).all(), "Negative transition prob"
        print("  transition_matrix: rows sum to 1, all non-negative ✓")

        # Regime statistics
        assert isinstance(result.regime_statistics, pd.DataFrame)
        assert set(result.regime_statistics.columns) >= {"mean_vol", "mean_return", "n_days"}
        assert result.regime_statistics["n_days"].sum() == expected_rows, (
            f"n_days sum mismatch: {result.regime_statistics['n_days'].sum()} "
            f"!= {expected_rows}"
        )
        print(f"  regime_statistics:\n{result.regime_statistics}")

        # Calm regime should have lower mean_vol than crisis
        row_labels = list(result.regime_statistics.index)
        if "calm" in row_labels and "crisis" in row_labels:
            calm_vol = result.regime_statistics.loc["calm", "mean_vol"]
            crisis_vol = result.regime_statistics.loc["crisis", "mean_vol"]
            assert calm_vol <= crisis_vol, (
                f"Calm vol ({calm_vol:.4f}) > crisis vol ({crisis_vol:.4f})"
            )
            print("  calm_vol <= crisis_vol (relabelling correct) ✓")

        # Average durations
        assert isinstance(result.avg_regime_duration_days, dict)
        assert len(result.avg_regime_duration_days) == 3
        print(f"  avg_regime_duration_days: {result.avg_regime_duration_days} ✓")

        # Current state
        assert 0 <= result.current_state < 3
        assert result.current_state_label in {"calm", "elevated", "crisis"}
        assert 0.0 <= result.current_state_probability <= 1.0
        print(
            f"  current_state='{result.current_state_label}' "
            f"(p={result.current_state_probability:.2%}) ✓"
        )

        # Scalar fields
        assert isinstance(result.fit_log_likelihood, float)
        assert isinstance(result.convergence_achieved, bool)
        print(f"  fit_log_likelihood={result.fit_log_likelihood:.2f}, "
              f"converged={result.convergence_achieved} ✓")

        # ── get_regime_correlation (mock DCC result) ─────────────────────
        # Build a minimal mock DCCGARCHResult-like object
        class _MockDCC:
            sector_names = ["Technology", "Financials", "Energy"]
            conditional_volatilities = pd.DataFrame(
                np.random.rand(expected_rows, 3) * 0.01,
                index=result.state_sequence.index,
                columns=sector_names,
            )
            conditional_correlations = np.stack([
                np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
                for _ in range(expected_rows)
            ])

        mock_dcc = _MockDCC()

        for lbl in ("calm", "elevated", "crisis"):
            corr = detector.get_regime_correlation(lbl, mock_dcc, result)
            assert corr.shape == (3, 3), f"Regime corr shape wrong for '{lbl}'"
            assert np.allclose(np.diag(corr.values), 1.0, atol=1e-9), (
                f"Diagonal not 1 for '{lbl}'"
            )
        print("  get_regime_correlation(): all 3 regimes, diagonal=1 ✓")

        # ── get_current_regime_correlation ───────────────────────────────
        curr_corr = detector.get_current_regime_correlation(mock_dcc, result)
        assert curr_corr.shape == (3, 3)
        assert np.allclose(np.diag(curr_corr.values), 1.0, atol=1e-9)
        print("  get_current_regime_correlation() ✓")

        # ── Invalid regime_label raises ValueError ───────────────────────
        try:
            detector.get_regime_correlation("panic", mock_dcc, result)
            raise AssertionError("Expected ValueError")
        except ValueError:
            pass
        print("  get_regime_correlation: invalid label raises ValueError ✓")

        # ── 4-feature config ─────────────────────────────────────────────
        config4 = RegimeConfig(
            n_states=2,
            n_init=3,
            n_iter=50,
            features=["rolling_vol", "mean_return", "vol_of_vol", "cross_sector_dispersion"],
        )
        result4 = MarketRegimeDetector(config4).fit(sector_returns)
        assert set(result4.state_sequence.unique()).issubset({0, 1})
        assert result4.current_state_label in {"calm", "crisis"}
        print("  4-feature / 2-state HMM ✓")

        print("\n✓ [MarketRegimeDetector] smoke test passed")

    _smoke_test()
