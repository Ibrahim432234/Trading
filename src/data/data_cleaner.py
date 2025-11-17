"""Data cleaning and validation module."""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("trading_system")


class DataCleaner:
    """Clean and validate OHLCV data."""

    @staticmethod
    def clean_ohlcv(df: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
        """
        Clean OHLCV data.

        Args:
            df: Raw OHLCV DataFrame
            fill_missing: Whether to fill missing values

        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        df = df.copy()

        # Remove duplicates
        initial_len = len(df)
        df = df[~df.index.duplicated(keep='first')]
        if len(df) < initial_len:
            logger.warning(f"Removed {initial_len - len(df)} duplicate rows")

        # Sort by timestamp
        df = df.sort_index()

        # Validate OHLCV logic (high >= low, etc.)
        df = DataCleaner._validate_ohlcv(df)

        # Handle missing values
        if fill_missing:
            df = DataCleaner._fill_missing_values(df)

        # Remove outliers
        df = DataCleaner._remove_outliers(df)

        logger.info(f"Cleaning complete. Final shape: {df.shape}")

        return df

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLCV relationships."""
        # Check high >= low
        invalid_hl = df['high'] < df['low']
        if invalid_hl.any():
            logger.warning(f"Found {invalid_hl.sum()} rows where high < low. Fixing...")
            df.loc[invalid_hl, ['high', 'low']] = df.loc[invalid_hl, ['low', 'high']].values

        # Check open/close within high/low range
        df['open'] = df[['open', 'high']].min(axis=1)
        df['open'] = df[['open', 'low']].max(axis=1)
        df['close'] = df[['close', 'high']].min(axis=1)
        df['close'] = df[['close', 'low']].max(axis=1)

        # Check for negative or zero prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            invalid = df[col] <= 0
            if invalid.any():
                logger.warning(f"Found {invalid.sum()} invalid {col} values. Removing rows...")
                df = df[~invalid]

        # Check for negative volume
        invalid_vol = df['volume'] < 0
        if invalid_vol.any():
            logger.warning(f"Found {invalid_vol.sum()} negative volume values. Setting to 0...")
            df.loc[invalid_vol, 'volume'] = 0

        return df

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values."""
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values. Filling...")

            # Forward fill for prices
            price_cols = ['open', 'high', 'low', 'close']
            df[price_cols] = df[price_cols].fillna(method='ffill')

            # Fill volume with 0
            df['volume'] = df['volume'].fillna(0)

            # Backward fill for any remaining
            df = df.fillna(method='bfill')

        return df

    @staticmethod
    def _remove_outliers(df: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
        """
        Remove extreme outliers using z-score method.

        Args:
            df: DataFrame
            z_threshold: Z-score threshold for outlier detection

        Returns:
            DataFrame with outliers removed
        """
        # Calculate returns for outlier detection
        returns = df['close'].pct_change()

        # Calculate z-scores
        z_scores = np.abs((returns - returns.mean()) / returns.std())

        # Flag outliers
        outliers = z_scores > z_threshold
        outlier_count = outliers.sum()

        if outlier_count > 0:
            logger.warning(f"Found {outlier_count} potential outliers (z-score > {z_threshold})")
            # Remove only extreme outliers
            df = df[~outliers]

        return df

    @staticmethod
    def resample_timeframe(
        df: pd.DataFrame,
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Resample data to different timeframe.

        Args:
            df: OHLCV DataFrame
            target_timeframe: Target timeframe (e.g., '4H', '1D')

        Returns:
            Resampled DataFrame
        """
        logger.info(f"Resampling to {target_timeframe}")

        resampled = df.resample(target_timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return resampled

    @staticmethod
    def get_data_quality_report(df: pd.DataFrame) -> dict:
        """
        Generate data quality report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with quality metrics
        """
        report = {
            'total_rows': len(df),
            'date_range': {
                'start': str(df.index.min()),
                'end': str(df.index.max()),
                'days': (df.index.max() - df.index.min()).days
            },
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.index.duplicated().sum(),
            'zero_volume_candles': (df['volume'] == 0).sum(),
            'price_stats': {
                'min': float(df['close'].min()),
                'max': float(df['close'].max()),
                'mean': float(df['close'].mean()),
                'std': float(df['close'].std())
            },
            'volume_stats': {
                'min': float(df['volume'].min()),
                'max': float(df['volume'].max()),
                'mean': float(df['volume'].mean())
            }
        }

        return report
