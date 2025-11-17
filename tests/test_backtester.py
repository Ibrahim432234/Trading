"""Tests for backtester."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.backtest.backtester import Backtester


def create_sample_data(n_rows=100):
    """Create sample OHLCV data."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='1H')

    # Create trending price data
    base_price = 100
    prices = [base_price]
    for i in range(1, n_rows):
        change = np.random.normal(0, 1)
        prices.append(prices[-1] + change)

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_rows)
    }, index=dates)

    return df


def create_simple_signals(n_rows=100):
    """Create simple buy-and-hold signals."""
    dates = pd.date_range(start='2023-01-01', periods=n_rows, freq='1H')
    signals = pd.Series([2] * n_rows, index=dates)  # All buy signals
    return signals


def test_backtester_initialization():
    """Test backtester initialization."""
    config = {
        'initial_capital': 10000,
        'costs': {'commission': 0.001, 'slippage': 0.0005},
        'position_size': 0.1,
        'risk_management': {
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_drawdown': 0.20,
            'max_positions': 1
        }
    }

    backtester = Backtester(config)

    assert backtester.initial_capital == 10000
    assert backtester.commission == 0.001
    assert backtester.position_size == 0.1


def test_backtest_run():
    """Test running a backtest."""
    config = {
        'initial_capital': 10000,
        'costs': {'commission': 0.001, 'slippage': 0.0005},
        'position_size': 0.1,
        'risk_management': {
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'max_drawdown': 0.20,
            'max_positions': 1
        }
    }

    backtester = Backtester(config)
    df = create_sample_data(100)
    signals = create_simple_signals(100)

    results = backtester.run(df, signals)

    assert 'final_equity' in results
    assert 'total_return' in results
    assert 'sharpe_ratio' in results
    assert 'max_drawdown' in results
    assert results['initial_capital'] == 10000


def test_backtest_metrics():
    """Test backtest metrics calculation."""
    config = {
        'initial_capital': 10000,
        'costs': {'commission': 0.001, 'slippage': 0.0005},
        'position_size': 0.1,
        'risk_management': {
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'max_drawdown': 0.20,
            'max_positions': 1
        }
    }

    backtester = Backtester(config)
    df = create_sample_data(100)
    signals = create_simple_signals(100)

    results = backtester.run(df, signals)

    # Check all required metrics are present
    required_metrics = [
        'total_return', 'cagr', 'sharpe_ratio', 'sortino_ratio',
        'max_drawdown', 'total_trades', 'win_rate'
    ]

    for metric in required_metrics:
        assert metric in results


def test_trade_log():
    """Test trade log generation."""
    config = {
        'initial_capital': 10000,
        'costs': {'commission': 0.001, 'slippage': 0.0005},
        'position_size': 0.1,
        'risk_management': {
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'max_drawdown': 0.20,
            'max_positions': 1
        }
    }

    backtester = Backtester(config)
    df = create_sample_data(100)

    # Create alternating signals
    signals = pd.Series([2 if i % 20 < 10 else 0 for i in range(100)], index=df.index)

    results = backtester.run(df, signals)
    trade_log = backtester.get_trade_log()

    if not trade_log.empty:
        assert 'entry_price' in trade_log.columns
        assert 'exit_price' in trade_log.columns
        assert 'pnl' in trade_log.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
