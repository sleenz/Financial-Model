"""Individual data source implementations with retry logic."""

import os
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from ..utils.logger import get_logger
from ..utils.helpers import chunk_list

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class DataSourceError(Exception):
    """Custom exception for data source errors."""
    pass


class RateLimitError(DataSourceError):
    """Exception for rate limit errors."""
    pass


class BaseDataSource(ABC):
    """Abstract base class for data sources."""

    def __init__(self, name: str):
        """
        Initialize the data source.

        Args:
            name: Name of the data source
        """
        self.name = name
        self._is_available = True

    @abstractmethod
    def fetch_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch price data for given tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with prices (index=dates, columns=tickers)
        """
        pass

    @property
    def is_available(self) -> bool:
        """Check if the data source is available."""
        return self._is_available

    def mark_unavailable(self):
        """Mark this source as unavailable."""
        self._is_available = False
        logger.warning(f"{self.name} marked as unavailable")


class YFinanceSource(BaseDataSource):
    """Yahoo Finance data source using yfinance library."""

    def __init__(self, batch_size: int = 5):
        """
        Initialize YFinance source.

        Args:
            batch_size: Number of tickers to fetch per batch
        """
        super().__init__("YFinance")
        self.batch_size = batch_size

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
        reraise=True,
    )
    def _fetch_batch(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch a batch of tickers with retry logic."""
        import yfinance as yf

        # Add random jitter to avoid rate limits
        time.sleep(random.uniform(0.1, 0.5))

        ticker_str = " ".join(tickers)
        data = yf.download(
            ticker_str,
            start=start_date,
            end=end_date,
            progress=False,
            threads=False,
            auto_adjust=True,
        )

        if data.empty:
            raise DataSourceError(f"No data returned for {tickers}")

        # Handle single ticker vs multiple tickers
        if len(tickers) == 1:
            if "Close" in data.columns:
                return data[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            if "Close" in data.columns.get_level_values(0):
                return data["Close"]

        return data

    def fetch_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch price data from Yahoo Finance.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with closing prices
        """
        logger.info(f"Fetching {len(tickers)} tickers from {self.name}")

        all_data = []
        batches = chunk_list(tickers, self.batch_size)

        for i, batch in enumerate(batches):
            try:
                logger.debug(f"Fetching batch {i+1}/{len(batches)}: {batch}")
                batch_data = self._fetch_batch(batch, start_date, end_date)
                all_data.append(batch_data)

                # Add delay between batches
                if i < len(batches) - 1:
                    time.sleep(random.uniform(0.5, 1.5))

            except Exception as e:
                logger.error(f"Error fetching batch {batch}: {e}")
                # Continue with remaining batches
                continue

        if not all_data:
            raise DataSourceError(f"Failed to fetch any data from {self.name}")

        # Combine all batches
        result = pd.concat(all_data, axis=1)
        logger.info(f"Successfully fetched {len(result.columns)} tickers from {self.name}")

        return result


class AlphaVantageSource(BaseDataSource):
    """Alpha Vantage data source."""

    def __init__(self, api_key: str = None):
        """
        Initialize Alpha Vantage source.

        Args:
            api_key: Alpha Vantage API key (or from env)
        """
        super().__init__("AlphaVantage")
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_KEY")
        self.base_url = "https://www.alphavantage.co/query"

        if not self.api_key or self.api_key == "your_alpha_vantage_key_here":
            logger.warning("Alpha Vantage API key not configured")
            self._is_available = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
        reraise=True,
    )
    def _fetch_single(self, ticker: str, outputsize: str = "full") -> pd.DataFrame:
        """Fetch data for a single ticker."""
        params = {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": ticker,
            "outputsize": outputsize,
            "apikey": self.api_key,
        }

        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Check for error messages
        if "Error Message" in data:
            raise DataSourceError(f"Alpha Vantage error: {data['Error Message']}")

        if "Note" in data:
            # Rate limit hit
            raise RateLimitError(f"Alpha Vantage rate limit: {data['Note']}")

        if "Time Series (Daily)" not in data:
            raise DataSourceError(f"Unexpected response format for {ticker}")

        # Parse time series data
        ts_data = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts_data, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Get adjusted close price
        close_col = "5. adjusted close"
        if close_col not in df.columns:
            close_col = "4. close"

        return df[[close_col]].astype(float).rename(columns={close_col: ticker})

    def fetch_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch price data from Alpha Vantage.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with closing prices
        """
        if not self._is_available:
            raise DataSourceError(f"{self.name} is not available (no API key)")

        logger.info(f"Fetching {len(tickers)} tickers from {self.name}")

        all_data = []

        for ticker in tickers:
            try:
                logger.debug(f"Fetching {ticker} from {self.name}")
                ticker_data = self._fetch_single(ticker)

                # Filter date range
                ticker_data = ticker_data[
                    (ticker_data.index >= pd.to_datetime(start_date)) &
                    (ticker_data.index <= pd.to_datetime(end_date))
                ]

                all_data.append(ticker_data)

                # Respect rate limits (5 calls per minute for free tier)
                time.sleep(12)  # 12 seconds between calls

            except RateLimitError as e:
                logger.warning(f"Rate limit hit for {self.name}: {e}")
                self.mark_unavailable()
                raise
            except Exception as e:
                logger.error(f"Error fetching {ticker} from {self.name}: {e}")
                continue

        if not all_data:
            raise DataSourceError(f"Failed to fetch any data from {self.name}")

        result = pd.concat(all_data, axis=1)
        logger.info(f"Successfully fetched {len(result.columns)} tickers from {self.name}")

        return result


class TwelveDataSource(BaseDataSource):
    """Twelve Data API source."""

    def __init__(self, api_key: str = None):
        """
        Initialize Twelve Data source.

        Args:
            api_key: Twelve Data API key (or from env)
        """
        super().__init__("TwelveData")
        self.api_key = api_key or os.getenv("TWELVE_DATA_KEY")
        self.base_url = "https://api.twelvedata.com/time_series"

        if not self.api_key or self.api_key == "your_twelve_data_key_here":
            logger.warning("Twelve Data API key not configured")
            self._is_available = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
        reraise=True,
    )
    def _fetch_single(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch data for a single ticker."""
        params = {
            "symbol": ticker,
            "interval": "1day",
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "apikey": self.api_key,
        }

        response = requests.get(self.base_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Check for errors
        if data.get("status") == "error":
            if "API" in data.get("message", "").upper():
                raise RateLimitError(f"Twelve Data rate limit: {data.get('message')}")
            raise DataSourceError(f"Twelve Data error: {data.get('message')}")

        if "values" not in data:
            raise DataSourceError(f"No data returned for {ticker}")

        # Parse values
        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").sort_index()
        df["close"] = df["close"].astype(float)

        return df[["close"]].rename(columns={"close": ticker})

    def fetch_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch price data from Twelve Data.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with closing prices
        """
        if not self._is_available:
            raise DataSourceError(f"{self.name} is not available (no API key)")

        logger.info(f"Fetching {len(tickers)} tickers from {self.name}")

        all_data = []

        for ticker in tickers:
            try:
                logger.debug(f"Fetching {ticker} from {self.name}")
                ticker_data = self._fetch_single(ticker, start_date, end_date)
                all_data.append(ticker_data)

                # Rate limiting (8 calls per minute for free tier)
                time.sleep(8)

            except RateLimitError as e:
                logger.warning(f"Rate limit hit for {self.name}: {e}")
                self.mark_unavailable()
                raise
            except Exception as e:
                logger.error(f"Error fetching {ticker} from {self.name}: {e}")
                continue

        if not all_data:
            raise DataSourceError(f"Failed to fetch any data from {self.name}")

        result = pd.concat(all_data, axis=1)
        logger.info(f"Successfully fetched {len(result.columns)} tickers from {self.name}")

        return result


class FMPSource(BaseDataSource):
    """Financial Modeling Prep data source."""

    def __init__(self, api_key: str = None):
        """
        Initialize FMP source.

        Args:
            api_key: FMP API key (or from env)
        """
        super().__init__("FMP")
        self.api_key = api_key or os.getenv("FMP_KEY")
        self.base_url = "https://financialmodelingprep.com/api/v3"

        if not self.api_key or self.api_key == "your_fmp_key_here":
            logger.warning("FMP API key not configured")
            self._is_available = False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
        reraise=True,
    )
    def _fetch_single(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch data for a single ticker."""
        url = f"{self.base_url}/historical-price-full/{ticker}"
        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "apikey": self.api_key,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Check for errors
        if isinstance(data, dict) and "Error Message" in data:
            raise DataSourceError(f"FMP error: {data['Error Message']}")

        if isinstance(data, dict) and "historical" not in data:
            raise DataSourceError(f"No historical data for {ticker}")

        # Parse historical data
        historical = data.get("historical", [])
        if not historical:
            raise DataSourceError(f"Empty historical data for {ticker}")

        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # Use adjusted close if available
        close_col = "adjClose" if "adjClose" in df.columns else "close"

        return df[[close_col]].astype(float).rename(columns={close_col: ticker})

    def fetch_prices(
        self,
        tickers: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Fetch price data from FMP.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with closing prices
        """
        if not self._is_available:
            raise DataSourceError(f"{self.name} is not available (no API key)")

        logger.info(f"Fetching {len(tickers)} tickers from {self.name}")

        all_data = []

        for ticker in tickers:
            try:
                logger.debug(f"Fetching {ticker} from {self.name}")
                ticker_data = self._fetch_single(ticker, start_date, end_date)
                all_data.append(ticker_data)

                # Rate limiting
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching {ticker} from {self.name}: {e}")
                continue

        if not all_data:
            raise DataSourceError(f"Failed to fetch any data from {self.name}")

        result = pd.concat(all_data, axis=1)
        logger.info(f"Successfully fetched {len(result.columns)} tickers from {self.name}")

        return result
