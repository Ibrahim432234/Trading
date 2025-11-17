"""Tests for data cleaner."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.data.data_cleaner import DataCleaner


def create_sample_data(n_rows=100):
    """Create sample OHLCV data."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='1H')
    data = {
        'open': np.random.uniform(100, 110, n_rows),
        'high': np.random.uniform(110, 120, n_rows),
        'low': np.random.uniform(90, 100, n_rows),
        'close': np.random.uniform(100, 110, n_rows),
        'volume': np.random.uniform(1000, 10000, n_rows)
    }
    df = pd.DataFrame(data, index=dates)
    return df


def test_clean_ohlcv_basic():
    """Test basic cleaning."""
    df = create_sample_data()
    cleaner = DataCleaner()

    cleaned = cleaner.clean_ohlcv(df)

    assert len(cleaned) > 0
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.duplicated().any()


def test_remove_duplicates():
    """Test duplicate removal."""
    df = create_sample_data(50)
    # Add duplicates
    df = pd.concat([df, df.iloc[:10]])

    cleaner = DataCleaner()
    cleaned = cleaner.clean_ohlcv(df)

    assert not cleaned.index.duplicated().any()


def test_validate_ohlcv():
    """Test OHLCV validation."""
    df = create_sample_data()
    # Introduce invalid data
    df.iloc[10, df.columns.get_loc('high')] = 50  # High < Low
    df.iloc[20, df.columns.get_loc('close')] = -10  # Negative price

    cleaner = DataCleaner()
    cleaned = cleaner.clean_ohlcv(df)

    # Check high >= low
    assert (cleaned['high'] >= cleaned['low']).all()
    # Check no negative prices
    assert (cleaned[['open', 'high', 'low', 'close']] > 0).all().all()


def test_fill_missing_values():
    """Test missing value filling."""
    df = create_sample_data()
    # Introduce missing values
    df.iloc[10:15, df.columns.get_loc('close')] = np.nan

    cleaner = DataCleaner()
    cleaned = cleaner.clean_ohlcv(df, fill_missing=True)

    assert not cleaned.isnull().any().any()


def test_data_quality_report():
    """Test data quality report generation."""
    df = create_sample_data()
    cleaner = DataCleaner()

    report = cleaner.get_data_quality_report(df)

    assert 'total_rows' in report
    assert 'date_range' in report
    assert 'price_stats' in report
    assert report['total_rows'] == len(df)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
