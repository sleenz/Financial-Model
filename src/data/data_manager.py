"""Data Manager with multi-source fallback and intelligent caching."""

from datetime import datetime
from typing import List, Optional, Union
import pandas as pd
from tqdm import tqdm

from .cache import DataCache
from .sources import (
    BaseDataSource,
    YFinanceSource,
    AlphaVantageSource,
    TwelveDataSource,
    FMPSource,
    DataSourceError,
    RateLimitError,
)
from .validators import DataValidator, quick_validate
from ..utils.logger import get_logger
from ..utils.helpers import validate_tickers, calculate_returns

logger = get_logger(__name__)


class DataManager:
    """
    Main data manager with multi-source fallback and caching.

    Handles data fetching from multiple sources with intelligent
    fallback when primary sources fail or hit rate limits.
    """

    def __init__(
        self,
        cache: DataCache = None,
        validator: DataValidator = None,
        enable_cache: bool = True,
        show_progress: bool = True,
    ):
        """
        Initialize the DataManager.

        Args:
            cache: Custom cache instance (default: creates new DataCache)
            validator: Custom validator instance (default: creates new DataValidator)
            enable_cache: Whether to enable caching
            show_progress: Whether to show progress indicators
        """
        self.cache = cache if cache else DataCache()
        self.validator = validator if validator else DataValidator()
        self.enable_cache = enable_cache
        self.show_progress = show_progress

        # Initialize data sources in priority order
        self.sources: List[BaseDataSource] = [
            YFinanceSource(),
            AlphaVantageSource(),
            TwelveDataSource(),
            FMPSource(),
        ]

        logger.info(f"DataManager initialized with {len(self.sources)} data sources")

    def get_price_data(
        self,
        tickers: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        validate: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch price data for given tickers with automatic fallback.

        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date (string or datetime)
            end_date: End date (string or datetime, default: today)
            validate: Whether to validate the data
            use_cache: Whether to use cache (if enabled)

        Returns:
            DataFrame with closing prices (index=dates, columns=tickers)

        Raises:
            DataSourceError: If all sources fail
        """
        # Normalize inputs
        tickers = validate_tickers(tickers)

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        logger.info(
            f"Fetching data for {len(tickers)} tickers from "
            f"{start_date.date()} to {end_date.date()}"
        )

        # Check cache first
        if self.enable_cache and use_cache:
            cache_key = self.cache._generate_key(
                "prices", tuple(sorted(tickers)),
                start_date.isoformat(), end_date.isoformat()
            )
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info("Returning cached data")
                return cached_data

        # Try each data source
        data = None
        last_error = None
        failed_sources = []

        for source in self.sources:
            if not source.is_available:
                logger.debug(f"Skipping unavailable source: {source.name}")
                continue

            try:
                logger.info(f"Attempting to fetch from {source.name}")

                if self.show_progress:
                    print(f"Fetching data from {source.name}...")

                data = source.fetch_prices(tickers, start_date, end_date)

                if data is not None and not data.empty:
                    logger.info(f"Successfully fetched data from {source.name}")
                    break

            except RateLimitError as e:
                logger.warning(f"{source.name} rate limited: {e}")
                failed_sources.append((source.name, "rate limited"))
                last_error = e
                continue

            except DataSourceError as e:
                logger.warning(f"{source.name} failed: {e}")
                failed_sources.append((source.name, str(e)))
                last_error = e
                continue

            except Exception as e:
                logger.error(f"Unexpected error from {source.name}: {e}")
                failed_sources.append((source.name, str(e)))
                last_error = e
                continue

        # Check if we got any data
        if data is None or data.empty:
            error_msg = f"All data sources failed. Tried: {failed_sources}"
            logger.error(error_msg)
            raise DataSourceError(error_msg)

        # Validate data
        if validate:
            data, report = self.validator.validate_price_data(data, tickers)

            if not report["valid"]:
                logger.warning(f"Data validation issues: {report['warnings']}")

            if report["removed_tickers"]:
                logger.warning(f"Removed tickers due to data issues: {report['removed_tickers']}")

        # Cache the result
        if self.enable_cache and use_cache:
            self.cache.set(data, cache_key, data_type="historical")

        return data

    def get_returns(
        self,
        tickers: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
        method: str = "simple",
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch returns data for given tickers.

        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date
            end_date: End date (default: today)
            method: Return calculation method ("simple" or "log")
            validate: Whether to validate data

        Returns:
            DataFrame with returns
        """
        # Fetch prices
        prices = self.get_price_data(tickers, start_date, end_date, validate)

        # Calculate returns
        returns = calculate_returns(prices, method)

        return returns

    def get_ohlcv_data(
        self,
        tickers: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None,
    ) -> dict:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.

        Args:
            tickers: Single ticker or list of tickers
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with DataFrames for each OHLCV field
        """
        import yfinance as yf

        tickers = validate_tickers(tickers)

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        logger.info(f"Fetching OHLCV data for {len(tickers)} tickers")

        # Use yfinance for OHLCV data
        ticker_str = " ".join(tickers)
        data = yf.download(
            ticker_str,
            start=start_date,
            end=end_date,
            progress=self.show_progress,
            auto_adjust=True,
        )

        if data.empty:
            raise DataSourceError("Failed to fetch OHLCV data")

        # Organize into dictionary
        result = {}
        fields = ["Open", "High", "Low", "Close", "Volume"]

        for field in fields:
            if len(tickers) == 1:
                if field in data.columns:
                    result[field.lower()] = data[[field]].rename(columns={field: tickers[0]})
            else:
                if field in data.columns.get_level_values(0):
                    result[field.lower()] = data[field]

        return result

    def get_current_prices(
        self,
        tickers: Union[str, List[str]],
    ) -> pd.Series:
        """
        Get current/latest prices for tickers.

        Args:
            tickers: Single ticker or list of tickers

        Returns:
            Series with current prices
        """
        import yfinance as yf

        tickers = validate_tickers(tickers)

        logger.info(f"Fetching current prices for {len(tickers)} tickers")

        prices = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                price = info.get("regularMarketPrice") or info.get("previousClose")
                if price:
                    prices[ticker] = price
            except Exception as e:
                logger.warning(f"Failed to get current price for {ticker}: {e}")

        return pd.Series(prices)

    def get_market_caps(
        self,
        tickers: Union[str, List[str]],
    ) -> pd.Series:
        """
        Get market capitalizations for tickers.

        Args:
            tickers: Single ticker or list of tickers

        Returns:
            Series with market caps
        """
        import yfinance as yf

        tickers = validate_tickers(tickers)

        logger.info(f"Fetching market caps for {len(tickers)} tickers")

        market_caps = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                market_cap = info.get("marketCap")
                if market_cap:
                    market_caps[ticker] = market_cap
            except Exception as e:
                logger.warning(f"Failed to get market cap for {ticker}: {e}")

        return pd.Series(market_caps)

    def get_sector_info(
        self,
        tickers: Union[str, List[str]],
    ) -> pd.DataFrame:
        """
        Get sector and industry information for tickers.

        Args:
            tickers: Single ticker or list of tickers

        Returns:
            DataFrame with sector and industry info
        """
        import yfinance as yf

        tickers = validate_tickers(tickers)

        logger.info(f"Fetching sector info for {len(tickers)} tickers")

        info_data = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                info_data.append({
                    "ticker": ticker,
                    "name": info.get("shortName", ""),
                    "sector": info.get("sector", "Unknown"),
                    "industry": info.get("industry", "Unknown"),
                })
            except Exception as e:
                logger.warning(f"Failed to get sector info for {ticker}: {e}")
                info_data.append({
                    "ticker": ticker,
                    "name": "",
                    "sector": "Unknown",
                    "industry": "Unknown",
                })

        return pd.DataFrame(info_data).set_index("ticker")

    def clear_cache(self) -> int:
        """
        Clear all cached data.

        Returns:
            Number of cache entries cleared
        """
        return self.cache.clear()

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return self.cache.get_stats()

    def get_available_sources(self) -> List[str]:
        """
        Get list of available data sources.

        Returns:
            List of available source names
        """
        return [s.name for s in self.sources if s.is_available]

    def set_source_priority(self, source_names: List[str]) -> None:
        """
        Set the priority order of data sources.

        Args:
            source_names: List of source names in priority order
        """
        source_map = {s.name: s for s in self.sources}
        new_order = []

        for name in source_names:
            if name in source_map:
                new_order.append(source_map[name])

        # Add any remaining sources
        for source in self.sources:
            if source not in new_order:
                new_order.append(source)

        self.sources = new_order
        logger.info(f"Source priority updated: {[s.name for s in self.sources]}")


def get_data(
    tickers: Union[str, List[str]],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to fetch price data.

    Args:
        tickers: Single ticker or list of tickers
        start_date: Start date
        end_date: End date (default: today)
        **kwargs: Additional arguments for DataManager

    Returns:
        DataFrame with closing prices
    """
    dm = DataManager(**kwargs)
    return dm.get_price_data(tickers, start_date, end_date)
