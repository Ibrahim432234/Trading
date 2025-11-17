"""Technical indicators for feature engineering."""

import pandas as pd
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger("trading_system")


class TechnicalIndicators:
    """Calculate technical indicators for trading signals."""

    @staticmethod
    def add_sma(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add Simple Moving Averages."""
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """Add Exponential Moving Averages."""
        for window in windows:
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """Add MACD indicator."""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        return df

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Add Bollinger Bands."""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()

        df['bb_upper'] = sma + (std * std_dev)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=period).mean()
        return df

    @staticmethod
    def add_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index."""
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        # Calculate DI+ and DI-
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(window=period).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        return df

    @staticmethod
    def add_roc(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """Add Rate of Change."""
        df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                                df['close'].shift(period)) * 100
        return df

    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Add Volume Weighted Average Price."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Add On-Balance Volume."""
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        df['obv'] = obv
        return df

    @staticmethod
    def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )

        df['cci'] = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return df

    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()

        df['williams_r'] = -100 * (high_max - df['close']) / (high_max - low_min)
        return df

    @staticmethod
    def add_momentum_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Add all momentum indicators based on config."""
        logger.info("Adding momentum indicators...")

        # SMA
        if 'sma_windows' in config:
            df = TechnicalIndicators.add_sma(df, config['sma_windows'])

        # EMA
        if 'ema_windows' in config:
            df = TechnicalIndicators.add_ema(df, config['ema_windows'])

        # RSI
        if 'rsi_period' in config:
            df = TechnicalIndicators.add_rsi(df, config['rsi_period'])

        # MACD
        if 'macd' in config:
            df = TechnicalIndicators.add_macd(df, *config['macd'])

        # Bollinger Bands
        if 'bbands_period' in config:
            df = TechnicalIndicators.add_bollinger_bands(df, config['bbands_period'])

        # ATR
        if 'atr_period' in config:
            df = TechnicalIndicators.add_atr(df, config['atr_period'])

        # Stochastic
        df = TechnicalIndicators.add_stochastic(df)

        # ADX
        df = TechnicalIndicators.add_adx(df)

        # ROC
        df = TechnicalIndicators.add_roc(df, 12)

        # CCI
        df = TechnicalIndicators.add_cci(df)

        # Williams %R
        df = TechnicalIndicators.add_williams_r(df)

        logger.info(f"Added indicators. Shape: {df.shape}")

        return df
