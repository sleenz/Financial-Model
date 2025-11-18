"""GARCH volatility modeling and forecasting."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try to import arch library
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logger.warning("arch library not available, using simplified GARCH implementation")


class GARCHModel:
    """
    GARCH volatility model for forecasting.

    Supports GARCH(1,1), EGARCH, and GJR-GARCH.
    """

    def __init__(
        self,
        returns: pd.Series,
        model_type: str = 'GARCH',
        p: int = 1,
        q: int = 1,
        dist: str = 'normal',
    ):
        """
        Initialize GARCH model.

        Args:
            returns: Series of returns
            model_type: 'GARCH', 'EGARCH', or 'GJR-GARCH'
            p: GARCH lag order
            q: ARCH lag order
            dist: Error distribution ('normal', 't', 'skewt')
        """
        self.returns = returns.dropna()
        self.model_type = model_type
        self.p = p
        self.q = q
        self.dist = dist
        self.fitted_model = None
        self.params = None

        logger.debug(f"Initialized {model_type}({p},{q}) model")

    def fit(self, update_freq: int = 0, disp: str = 'off') -> Dict:
        """
        Fit the GARCH model.

        Args:
            update_freq: Display frequency during optimization
            disp: Display option ('off', 'on', 'final')

        Returns:
            Dictionary with fitted parameters
        """
        if ARCH_AVAILABLE:
            return self._fit_arch(update_freq, disp)
        else:
            return self._fit_simple()

    def _fit_arch(self, update_freq: int, disp: str) -> Dict:
        """Fit using arch library."""
        # Scale returns for numerical stability
        scale = 100
        scaled_returns = self.returns * scale

        # Create model
        if self.model_type == 'GARCH':
            model = arch_model(
                scaled_returns,
                vol='Garch',
                p=self.p,
                q=self.q,
                dist=self.dist
            )
        elif self.model_type == 'EGARCH':
            model = arch_model(
                scaled_returns,
                vol='EGARCH',
                p=self.p,
                q=self.q,
                dist=self.dist
            )
        elif self.model_type == 'GJR-GARCH':
            model = arch_model(
                scaled_returns,
                vol='Garch',
                p=self.p,
                o=1,  # GJR term
                q=self.q,
                dist=self.dist
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Fit model
        self.fitted_model = model.fit(
            update_freq=update_freq,
            disp=disp
        )

        # Extract parameters
        self.params = {
            'omega': self.fitted_model.params.get('omega', 0) / (scale ** 2),
            'alpha': self.fitted_model.params.get('alpha[1]', 0),
            'beta': self.fitted_model.params.get('beta[1]', 0),
        }

        if self.model_type == 'GJR-GARCH':
            self.params['gamma'] = self.fitted_model.params.get('gamma[1]', 0)

        # Rescale conditional volatility
        self.conditional_vol = self.fitted_model.conditional_volatility / scale

        return {
            'params': self.params,
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'loglikelihood': self.fitted_model.loglikelihood,
        }

    def _fit_simple(self) -> Dict:
        """Simple GARCH(1,1) fitting without arch library."""
        returns = self.returns.values

        def garch_likelihood(params):
            omega, alpha, beta = params

            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10

            n = len(returns)
            sigma2 = np.zeros(n)
            sigma2[0] = np.var(returns)

            for t in range(1, n):
                sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]

            # Negative log-likelihood
            ll = 0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
            return ll

        # Initial guess
        var = np.var(returns)
        x0 = [var * 0.05, 0.05, 0.90]

        # Bounds
        bounds = [(1e-10, None), (0, 1), (0, 1)]

        # Optimize
        result = minimize(
            garch_likelihood,
            x0,
            method='L-BFGS-B',
            bounds=bounds
        )

        self.params = {
            'omega': result.x[0],
            'alpha': result.x[1],
            'beta': result.x[2],
        }

        # Calculate conditional volatility
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)

        for t in range(1, n):
            sigma2[t] = (
                self.params['omega'] +
                self.params['alpha'] * returns[t-1]**2 +
                self.params['beta'] * sigma2[t-1]
            )

        self.conditional_vol = pd.Series(
            np.sqrt(sigma2),
            index=self.returns.index
        )

        return {
            'params': self.params,
            'loglikelihood': -result.fun,
        }

    def forecast(
        self,
        horizon: int = 30,
        method: str = 'analytic'
    ) -> pd.DataFrame:
        """
        Forecast volatility.

        Args:
            horizon: Forecast horizon (days)
            method: 'analytic' or 'simulation'

        Returns:
            DataFrame with forecasts
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if ARCH_AVAILABLE and self.fitted_model is not None:
            return self._forecast_arch(horizon, method)
        else:
            return self._forecast_simple(horizon)

    def _forecast_arch(self, horizon: int, method: str) -> pd.DataFrame:
        """Forecast using arch library."""
        scale = 100
        forecast = self.fitted_model.forecast(horizon=horizon, method=method)

        # Rescale variance forecasts
        variance_forecast = forecast.variance.iloc[-1] / (scale ** 2)
        volatility_forecast = np.sqrt(variance_forecast)

        return pd.DataFrame({
            'variance': variance_forecast.values,
            'volatility': volatility_forecast.values,
            'annualized_vol': volatility_forecast.values * np.sqrt(252)
        }, index=range(1, horizon + 1))

    def _forecast_simple(self, horizon: int) -> pd.DataFrame:
        """Simple analytic forecast."""
        omega = self.params['omega']
        alpha = self.params['alpha']
        beta = self.params['beta']

        # Unconditional variance
        unconditional_var = omega / (1 - alpha - beta)

        # Last variance
        last_var = self.conditional_vol.iloc[-1] ** 2

        # Multi-step forecast
        forecasts = []
        var_t = last_var

        for h in range(1, horizon + 1):
            var_t = omega + (alpha + beta) * var_t
            forecasts.append({
                'variance': var_t,
                'volatility': np.sqrt(var_t),
                'annualized_vol': np.sqrt(var_t * 252)
            })

        return pd.DataFrame(forecasts, index=range(1, horizon + 1))

    def get_conditional_volatility(self, annualize: bool = True) -> pd.Series:
        """
        Get fitted conditional volatility.

        Args:
            annualize: Whether to annualize

        Returns:
            Conditional volatility series
        """
        if self.conditional_vol is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if annualize:
            return self.conditional_vol * np.sqrt(252)
        return self.conditional_vol

    def persistence(self) -> float:
        """
        Calculate volatility persistence.

        Returns:
            Persistence (alpha + beta)
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.params['alpha'] + self.params['beta']

    def half_life(self) -> float:
        """
        Calculate half-life of volatility shocks.

        Returns:
            Half-life in periods
        """
        persistence = self.persistence()
        if persistence >= 1:
            return np.inf
        return np.log(0.5) / np.log(persistence)

    def unconditional_volatility(self, annualize: bool = True) -> float:
        """
        Calculate unconditional (long-run) volatility.

        Args:
            annualize: Whether to annualize

        Returns:
            Unconditional volatility
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        omega = self.params['omega']
        persistence = self.persistence()

        if persistence >= 1:
            return np.inf

        unconditional_var = omega / (1 - persistence)
        vol = np.sqrt(unconditional_var)

        if annualize:
            vol = vol * np.sqrt(252)

        return vol

    def summary(self) -> Dict:
        """
        Get model summary.

        Returns:
            Dictionary with model summary
        """
        if self.params is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return {
            'model_type': self.model_type,
            'parameters': self.params,
            'persistence': self.persistence(),
            'half_life': self.half_life(),
            'unconditional_vol': self.unconditional_volatility(),
        }


class MultiAssetGARCH:
    """
    Fit GARCH models for multiple assets.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        model_type: str = 'GARCH',
        p: int = 1,
        q: int = 1,
    ):
        """
        Initialize multi-asset GARCH.

        Args:
            returns: DataFrame of returns
            model_type: GARCH model type
            p: GARCH lag order
            q: ARCH lag order
        """
        self.returns = returns
        self.model_type = model_type
        self.p = p
        self.q = q
        self.models = {}

    def fit_all(self, disp: str = 'off') -> Dict:
        """
        Fit GARCH model for each asset.

        Args:
            disp: Display option

        Returns:
            Dictionary with all model summaries
        """
        summaries = {}

        for col in self.returns.columns:
            logger.info(f"Fitting {self.model_type} for {col}")

            model = GARCHModel(
                self.returns[col],
                self.model_type,
                self.p,
                self.q
            )

            model.fit(disp=disp)
            self.models[col] = model
            summaries[col] = model.summary()

        return summaries

    def forecast_all(self, horizon: int = 30) -> pd.DataFrame:
        """
        Forecast volatility for all assets.

        Args:
            horizon: Forecast horizon

        Returns:
            DataFrame with forecasts for all assets
        """
        if not self.models:
            raise ValueError("Models not fitted. Call fit_all() first.")

        forecasts = {}

        for col, model in self.models.items():
            forecast = model.forecast(horizon)
            forecasts[col] = forecast['annualized_vol']

        return pd.DataFrame(forecasts)

    def get_conditional_volatilities(self, annualize: bool = True) -> pd.DataFrame:
        """
        Get conditional volatilities for all assets.

        Args:
            annualize: Whether to annualize

        Returns:
            DataFrame with conditional volatilities
        """
        if not self.models:
            raise ValueError("Models not fitted. Call fit_all() first.")

        vols = {}
        for col, model in self.models.items():
            vols[col] = model.get_conditional_volatility(annualize)

        return pd.DataFrame(vols)


def forecast_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    horizon: int = 30,
    model_type: str = 'GARCH',
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Convenience function to forecast volatility.

    Args:
        returns: Return data
        horizon: Forecast horizon
        model_type: GARCH model type

    Returns:
        Volatility forecasts
    """
    if isinstance(returns, pd.Series):
        model = GARCHModel(returns, model_type)
        model.fit()
        return model.forecast(horizon)
    else:
        multi = MultiAssetGARCH(returns, model_type)
        multi.fit_all()
        return multi.forecast_all(horizon)


def ewma_volatility(
    returns: pd.DataFrame,
    decay: float = 0.94,
    annualize: bool = True
) -> pd.DataFrame:
    """
    Calculate EWMA (RiskMetrics) volatility.

    Args:
        returns: Return data
        decay: Decay factor (lambda)
        annualize: Whether to annualize

    Returns:
        EWMA volatility
    """
    # Initial variance
    var = returns.ewm(alpha=1-decay, adjust=False).var()

    vol = np.sqrt(var)

    if annualize:
        vol = vol * np.sqrt(252)

    return vol
