"""Tests for feature engineering."""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.features.technical_indicators import TechnicalIndicators
from src.features.feature_engineer import FeatureEngineer


def create_sample_data(n_rows=200):
    """Create sample OHLCV data."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='1H')

    base_price = 100
    prices = [base_price]
    for i in range(1, n_rows):
        change = np.random.normal(0, 1)
        prices.append(max(prices[-1] + change, 1))  # Ensure positive

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_rows)
    }, index=dates)

    return df


def test_sma():
    """Test SMA calculation."""
    df = create_sample_data()
    df = TechnicalIndicators.add_sma(df, [10, 20])

    assert 'sma_10' in df.columns
    assert 'sma_20' in df.columns
    assert df['sma_10'].notna().sum() > 0


def test_ema():
    """Test EMA calculation."""
    df = create_sample_data()
    df = TechnicalIndicators.add_ema(df, [12, 26])

    assert 'ema_12' in df.columns
    assert 'ema_26' in df.columns


def test_rsi():
    """Test RSI calculation."""
    df = create_sample_data()
    df = TechnicalIndicators.add_rsi(df, period=14)

    assert 'rsi' in df.columns
    # RSI should be between 0 and 100
    valid_rsi = df['rsi'].dropna()
    assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all()


def test_macd():
    """Test MACD calculation."""
    df = create_sample_data()
    df = TechnicalIndicators.add_macd(df)

    assert 'macd' in df.columns
    assert 'macd_signal' in df.columns
    assert 'macd_diff' in df.columns


def test_bollinger_bands():
    """Test Bollinger Bands calculation."""
    df = create_sample_data()
    df = TechnicalIndicators.add_bollinger_bands(df)

    assert 'bb_upper' in df.columns
    assert 'bb_middle' in df.columns
    assert 'bb_lower' in df.columns

    # Upper should be > Middle > Lower
    valid_rows = df[['bb_upper', 'bb_middle', 'bb_lower']].dropna()
    assert (valid_rows['bb_upper'] >= valid_rows['bb_middle']).all()
    assert (valid_rows['bb_middle'] >= valid_rows['bb_lower']).all()


def test_feature_engineer():
    """Test complete feature engineering."""
    df = create_sample_data()

    config = {
        'indicators': {
            'sma_windows': [20, 50],
            'ema_windows': [12, 26],
            'rsi_period': 14,
            'macd': [12, 26, 9],
            'bbands_period': 20,
            'atr_period': 14
        },
        'time_features': True,
        'lagged_returns': [1, 2, 3]
    }

    engineer = FeatureEngineer(config)
    df_features = engineer.create_features(df)

    # Check that features were added
    assert len(df_features.columns) > len(df.columns)
    assert 'sma_20' in df_features.columns
    assert 'rsi' in df_features.columns


def test_label_creation():
    """Test label creation."""
    df = create_sample_data()

    config = {'indicators': {}}
    engineer = FeatureEngineer(config)

    df_labeled = engineer.create_labels(
        df,
        label_type='classification',
        horizons=[1, 5],
        threshold=0.02
    )

    assert 'target_1' in df_labeled.columns
    assert 'target_5' in df_labeled.columns

    # Check labels are in valid range
    valid_labels = df_labeled['target_1'].dropna()
    assert valid_labels.isin([0, 1, 2]).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
