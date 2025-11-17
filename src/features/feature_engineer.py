"""Feature engineering pipeline."""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from .technical_indicators import TechnicalIndicators

logger = logging.getLogger("trading_system")


class FeatureEngineer:
    """Engineer features for machine learning."""

    def __init__(self, config: dict):
        """
        Initialize feature engineer.

        Args:
            config: Feature engineering configuration
        """
        self.config = config

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features.

        Args:
            df: OHLCV DataFrame

        Returns:
            DataFrame with features
        """
        logger.info("Starting feature engineering...")
        df = df.copy()

        # Technical indicators
        df = TechnicalIndicators.add_momentum_indicators(
            df,
            self.config.get('indicators', {})
        )

        # Price-based features
        df = self._add_price_features(df)

        # Volatility features
        df = self._add_volatility_features(df)

        # Volume features
        df = self._add_volume_features(df)

        # Time features
        if self.config.get('time_features', True):
            df = self._add_time_features(df)

        # Lagged returns
        if 'lagged_returns' in self.config:
            df = self._add_lagged_returns(df, self.config['lagged_returns'])

        # Pattern features
        df = self._add_pattern_features(df)

        logger.info(f"Feature engineering complete. Shape: {df.shape}")

        return df

    def create_labels(
        self,
        df: pd.DataFrame,
        label_type: str = "classification",
        horizons: List[int] = [1, 5, 20],
        threshold: float = 0.02
    ) -> pd.DataFrame:
        """
        Create labels for supervised learning.

        Args:
            df: DataFrame with features
            label_type: 'classification' or 'regression'
            horizons: Future periods to predict
            threshold: Threshold for classification (e.g., 0.02 = 2%)

        Returns:
            DataFrame with labels
        """
        logger.info(f"Creating {label_type} labels for horizons: {horizons}")

        for horizon in horizons:
            future_return = df['close'].shift(-horizon) / df['close'] - 1

            if label_type == "classification":
                # 0: down, 1: neutral, 2: up
                df[f'target_{horizon}'] = pd.cut(
                    future_return,
                    bins=[-np.inf, -threshold, threshold, np.inf],
                    labels=[0, 1, 2]
                ).astype(int)

            elif label_type == "regression":
                df[f'target_{horizon}'] = future_return

        return df

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price ranges
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['close_open_range'] = (df['close'] - df['open']) / df['open']

        # Rolling quantiles
        for window in [20, 50]:
            df[f'close_quantile_{window}'] = df['close'].rolling(window).apply(
                lambda x: pd.Series(x).rank().iloc[-1] / len(x)
            )

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        for window in [10, 20, 50]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'volatility_{window}_mean'] = df[f'volatility_{window}'].rolling(20).mean()

        # Parkinson volatility (using high-low)
        df['parkinson_volatility'] = np.sqrt(
            np.log(df['high'] / df['low']) ** 2 / (4 * np.log(2))
        )

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume features."""
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # Price-volume correlation
        for window in [10, 20]:
            df[f'price_volume_corr_{window}'] = df['close'].rolling(window).corr(
                df['volume']
            )

        # Money flow
        df['money_flow'] = df['close'] * df['volume']
        df['money_flow_ratio'] = (
            df['money_flow'].rolling(20).mean() /
            df['money_flow'].rolling(50).mean()
        )

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # Cyclical encoding for hour and day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _add_lagged_returns(self, df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
        """Add lagged returns."""
        for lag in lags:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)

        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern features."""
        # Body size
        df['body_size'] = abs(df['close'] - df['open']) / df['open']

        # Upper/lower shadows
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']

        # Doji pattern (small body)
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)

        # Hammer pattern
        df['is_hammer'] = (
            (df['lower_shadow'] > 2 * df['body_size']) &
            (df['upper_shadow'] < df['body_size'])
        ).astype(int)

        # Engulfing pattern
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)

        return df

    def prepare_ml_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'target_1',
        drop_na: bool = True
    ) -> tuple:
        """
        Prepare data for ML training.

        Args:
            df: DataFrame with features and labels
            target_col: Target column name
            drop_na: Whether to drop NaN values

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing ML data...")

        # Remove rows with missing target
        df = df.dropna(subset=[target_col])

        # Define feature columns (exclude target and OHLCV)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume'] + \
                       [col for col in df.columns if col.startswith('target_')]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols]
        y = df[target_col]

        if drop_na:
            # Drop rows with any NaN in features
            valid_idx = X.notna().all(axis=1)
            X = X[valid_idx]
            y = y[valid_idx]

            logger.info(f"Dropped {(~valid_idx).sum()} rows with NaN values")

        logger.info(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")

        return X, y, feature_cols

    @staticmethod
    def get_feature_groups() -> dict:
        """
        Get feature groups for analysis.

        Returns:
            Dictionary mapping group names to feature prefixes
        """
        return {
            'momentum': ['sma_', 'ema_', 'rsi', 'macd', 'roc'],
            'volatility': ['bb_', 'atr', 'volatility', 'parkinson'],
            'volume': ['volume', 'obv', 'money_flow', 'vwap'],
            'oscillators': ['stoch_', 'cci', 'williams_r', 'adx'],
            'price': ['returns', 'log_returns', 'range', 'quantile'],
            'time': ['hour', 'day_', 'month', 'quarter'],
            'patterns': ['body_size', 'shadow', 'doji', 'hammer', 'engulfing']
        }
