"""Tests for data infrastructure (Phase 1)."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.cache import DataCache
from src.data.validators import DataValidator, DataValidationError
from src.utils.helpers import (
    validate_tickers,
    format_currency,
    format_percentage,
    calculate_date_range,
    calculate_returns,
    annualize_returns,
    annualize_volatility,
    chunk_list,
)


class TestDataCache:
    """Tests for DataCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DataCache(cache_dir=self.temp_dir, ttl_intraday=60, ttl_historical=120)

    def teardown_method(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_set_and_get(self):
        """Test basic set and get operations."""
        data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        key = self.cache.set(data, key='test_key')

        retrieved = self.cache.get('test_key')
        assert retrieved is not None
        pd.testing.assert_frame_equal(data, retrieved)

    def test_cache_expiration(self):
        """Test that cache expires after TTL."""
        import time

        # Use very short TTL
        cache = DataCache(cache_dir=self.temp_dir, ttl_historical=1)
        data = {'value': 42}
        cache.set(data, key='expire_test', ttl=1)

        # Should be available immediately
        assert cache.get('expire_test') is not None

        # Wait for expiration
        time.sleep(1.5)
        assert cache.get('expire_test') is None

    def test_cache_delete(self):
        """Test cache deletion."""
        self.cache.set({'data': 1}, key='delete_test')
        assert self.cache.get('delete_test') is not None

        self.cache.delete('delete_test')
        assert self.cache.get('delete_test') is None

    def test_cache_clear(self):
        """Test clearing all cache."""
        self.cache.set({'data': 1}, key='key1')
        self.cache.set({'data': 2}, key='key2')

        count = self.cache.clear()
        assert count == 2
        assert self.cache.get('key1') is None
        assert self.cache.get('key2') is None

    def test_cache_stats(self):
        """Test cache statistics."""
        self.cache.set({'data': 1}, key='stat_test')
        stats = self.cache.get_stats()

        assert 'total_files' in stats
        assert stats['total_files'] >= 1

    def test_generate_key(self):
        """Test cache key generation."""
        key1 = self.cache._generate_key('arg1', 'arg2', kwarg1='value1')
        key2 = self.cache._generate_key('arg1', 'arg2', kwarg1='value1')
        key3 = self.cache._generate_key('arg1', 'arg2', kwarg1='value2')

        assert key1 == key2  # Same args = same key
        assert key1 != key3  # Different args = different key


class TestDataValidator:
    """Tests for DataValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = DataValidator()

    def test_validate_clean_data(self):
        """Test validation of clean data."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'AAPL': np.random.uniform(100, 200, 100),
            'GOOGL': np.random.uniform(1000, 2000, 100),
        }, index=dates)

        cleaned, report = self.validator.validate_price_data(data)

        assert report['valid']
        assert len(report['errors']) == 0
        assert len(cleaned.columns) == 2

    def test_validate_missing_values(self):
        """Test validation with missing values."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'AAPL': np.random.uniform(100, 200, 100),
        }, index=dates)

        # Add some missing values (within tolerance)
        data.iloc[5] = np.nan
        data.iloc[10] = np.nan

        cleaned, report = self.validator.validate_price_data(data)

        assert report['valid']
        assert 'AAPL' in report['filled_missing']
        assert not cleaned['AAPL'].isna().any()

    def test_validate_too_many_missing(self):
        """Test validation with too many missing values."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'AAPL': np.random.uniform(100, 200, 100),
        }, index=dates)

        # Add too many missing values (> 5%)
        data.iloc[:10] = np.nan

        cleaned, report = self.validator.validate_price_data(data)

        assert not report['valid']
        assert 'AAPL' in report['removed_tickers']

    def test_validate_negative_prices(self):
        """Test validation catches negative prices."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'AAPL': np.random.uniform(100, 200, 100),
        }, index=dates)

        # Add negative price
        data.iloc[50, 0] = -100

        cleaned, report = self.validator.validate_price_data(data)

        assert not report['valid']
        assert 'AAPL' in report['removed_tickers']

    def test_validate_insufficient_data(self):
        """Test validation with insufficient data points."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'AAPL': np.random.uniform(100, 200, 10),
        }, index=dates)

        cleaned, report = self.validator.validate_price_data(data)

        assert not report['valid']

    def test_check_date_range(self):
        """Test date range coverage check."""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'AAPL': np.random.uniform(100, 200, 100),
        }, index=dates)

        start = datetime(2020, 1, 1)
        end = datetime(2020, 4, 10)

        is_valid, msg = self.validator.check_date_range(data, start, end)
        assert is_valid

    def test_validate_returns(self):
        """Test return data validation."""
        returns = pd.DataFrame({
            'AAPL': np.random.normal(0.001, 0.02, 100),
        })

        # Add extreme return
        returns.iloc[50, 0] = 2.0  # 200% return

        cleaned, report = self.validator.validate_returns(returns)

        assert 'AAPL' in report['clipped_values']
        assert cleaned['AAPL'].iloc[50] == 1.0  # Clipped to 100%


class TestHelperFunctions:
    """Tests for utility helper functions."""

    def test_validate_tickers_list(self):
        """Test ticker validation with list input."""
        tickers = validate_tickers(['AAPL', 'GOOGL', 'MSFT'])
        assert tickers == ['AAPL', 'GOOGL', 'MSFT']

    def test_validate_tickers_string(self):
        """Test ticker validation with string input."""
        tickers = validate_tickers('AAPL, GOOGL, MSFT')
        assert tickers == ['AAPL', 'GOOGL', 'MSFT']

    def test_validate_tickers_case(self):
        """Test ticker validation normalizes case."""
        tickers = validate_tickers(['aapl', 'googl'])
        assert tickers == ['AAPL', 'GOOGL']

    def test_validate_tickers_invalid(self):
        """Test ticker validation rejects invalid tickers."""
        with pytest.raises(ValueError):
            validate_tickers(['AAPL', 'invalid@ticker'])

    def test_validate_tickers_empty(self):
        """Test ticker validation rejects empty input."""
        with pytest.raises(ValueError):
            validate_tickers('')

    def test_format_currency(self):
        """Test currency formatting."""
        assert format_currency(1234.56) == '$1,234.56'
        assert format_currency(-1234.56) == '-$1,234.56'
        assert format_currency(0) == '$0.00'

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.1523) == '15.23%'
        assert format_percentage(-0.05) == '-5.00%'
        assert format_percentage(1.0) == '100.00%'

    def test_calculate_date_range(self):
        """Test date range calculation."""
        end = datetime(2024, 1, 1)
        start, end_date = calculate_date_range('1y', end)

        assert end_date == end
        assert (end - start).days == 365

    def test_calculate_date_range_invalid(self):
        """Test date range calculation with invalid period."""
        with pytest.raises(ValueError):
            calculate_date_range('invalid')

    def test_calculate_returns_simple(self):
        """Test simple return calculation."""
        prices = pd.DataFrame({
            'A': [100, 110, 105],
        })
        returns = calculate_returns(prices, method='simple')

        assert len(returns) == 2
        assert abs(returns['A'].iloc[0] - 0.10) < 0.001
        assert abs(returns['A'].iloc[1] - (-0.0455)) < 0.001

    def test_calculate_returns_log(self):
        """Test log return calculation."""
        prices = pd.DataFrame({
            'A': [100, 110, 105],
        })
        returns = calculate_returns(prices, method='log')

        assert len(returns) == 2
        expected = np.log(110/100)
        assert abs(returns['A'].iloc[0] - expected) < 0.001

    def test_annualize_volatility(self):
        """Test volatility annualization."""
        daily_returns = pd.Series(np.random.normal(0, 0.01, 252))
        annual_vol = annualize_volatility(daily_returns)

        # Should be roughly sqrt(252) * daily_std
        expected = daily_returns.std() * np.sqrt(252)
        assert abs(annual_vol - expected) < 0.001

    def test_chunk_list(self):
        """Test list chunking."""
        lst = [1, 2, 3, 4, 5, 6, 7]
        chunks = chunk_list(lst, 3)

        assert len(chunks) == 3
        assert chunks[0] == [1, 2, 3]
        assert chunks[1] == [4, 5, 6]
        assert chunks[2] == [7]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
