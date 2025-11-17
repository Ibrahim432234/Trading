"""Trading strategies combining ML signals with technical analysis."""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("trading_system")


class TradingStrategy:
    """Base class for trading strategies."""

    def __init__(self, config: dict):
        """Initialize strategy."""
        self.config = config

    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_predictions: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Generate trading signals.

        Args:
            df: DataFrame with OHLCV and features
            ml_predictions: ML model predictions (0=sell, 1=hold, 2=buy)

        Returns:
            Series with signals (0=sell, 1=hold, 2=buy)
        """
        raise NotImplementedError


class MomentumMLStrategy(TradingStrategy):
    """Momentum strategy combined with ML signals."""

    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_predictions: Optional[pd.Series] = None
    ) -> pd.Series:
        """Generate momentum + ML signals."""
        logger.info("Generating momentum ML signals...")

        signals = pd.Series(1, index=df.index)  # Default: hold

        # Technical momentum
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            technical_buy = (df['sma_20'] > df['sma_50']) & (df['close'] > df['sma_20'])
            technical_sell = (df['sma_20'] < df['sma_50']) & (df['close'] < df['sma_20'])

            # Combine with ML if available
            if ml_predictions is not None:
                # Align indices
                common_idx = signals.index.intersection(ml_predictions.index)
                signals = signals.loc[common_idx]
                ml_predictions = ml_predictions.loc[common_idx]
                technical_buy = technical_buy.loc[common_idx]
                technical_sell = technical_sell.loc[common_idx]

                # Buy: Both ML and technical agree on buy
                signals[technical_buy & (ml_predictions == 2)] = 2

                # Sell: Both ML and technical agree on sell
                signals[technical_sell & (ml_predictions == 0)] = 0
            else:
                signals[technical_buy] = 2
                signals[technical_sell] = 0

        elif ml_predictions is not None:
            # Use ML predictions only
            common_idx = signals.index.intersection(ml_predictions.index)
            signals.loc[common_idx] = ml_predictions.loc[common_idx]

        logger.info(f"Generated {(signals == 2).sum()} buy and {(signals == 0).sum()} sell signals")

        return signals


class MeanReversionMLStrategy(TradingStrategy):
    """Mean reversion strategy combined with ML signals."""

    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_predictions: Optional[pd.Series] = None
    ) -> pd.Series:
        """Generate mean reversion + ML signals."""
        logger.info("Generating mean reversion ML signals...")

        signals = pd.Series(1, index=df.index)  # Default: hold

        # RSI-based mean reversion
        if 'rsi' in df.columns:
            oversold = df['rsi'] < 30
            overbought = df['rsi'] > 70

            # Bollinger Bands
            if 'bb_position' in df.columns:
                oversold = oversold & (df['bb_position'] < 0.1)
                overbought = overbought & (df['bb_position'] > 0.9)

            # Combine with ML if available
            if ml_predictions is not None:
                # Align indices
                common_idx = signals.index.intersection(ml_predictions.index)
                signals = signals.loc[common_idx]
                ml_predictions = ml_predictions.loc[common_idx]
                oversold = oversold.loc[common_idx]
                overbought = overbought.loc[common_idx]

                # Buy: Oversold and ML confirms
                signals[oversold & (ml_predictions >= 1)] = 2

                # Sell: Overbought and ML confirms
                signals[overbought & (ml_predictions <= 1)] = 0
            else:
                signals[oversold] = 2
                signals[overbought] = 0

        elif ml_predictions is not None:
            # Use ML predictions only
            common_idx = signals.index.intersection(ml_predictions.index)
            signals.loc[common_idx] = ml_predictions.loc[common_idx]

        logger.info(f"Generated {(signals == 2).sum()} buy and {(signals == 0).sum()} sell signals")

        return signals


class VolatilityBreakoutMLStrategy(TradingStrategy):
    """Volatility breakout strategy combined with ML signals."""

    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_predictions: Optional[pd.Series] = None
    ) -> pd.Series:
        """Generate volatility breakout + ML signals."""
        logger.info("Generating volatility breakout ML signals...")

        signals = pd.Series(1, index=df.index)  # Default: hold

        # Bollinger Band breakouts
        if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
            # Breakout above upper band
            breakout_up = (df['close'] > df['bb_upper']) & (df['bb_width'] > df['bb_width'].rolling(20).mean())

            # Breakout below lower band
            breakout_down = (df['close'] < df['bb_lower']) & (df['bb_width'] > df['bb_width'].rolling(20).mean())

            # ADX filter (trend strength)
            if 'adx' in df.columns:
                strong_trend = df['adx'] > 25
                breakout_up = breakout_up & strong_trend
                breakout_down = breakout_down & strong_trend

            # Combine with ML if available
            if ml_predictions is not None:
                # Align indices
                common_idx = signals.index.intersection(ml_predictions.index)
                signals = signals.loc[common_idx]
                ml_predictions = ml_predictions.loc[common_idx]
                breakout_up = breakout_up.loc[common_idx]
                breakout_down = breakout_down.loc[common_idx]

                # Buy: Breakout up and ML confirms
                signals[breakout_up & (ml_predictions == 2)] = 2

                # Sell: Breakout down and ML confirms
                signals[breakout_down & (ml_predictions == 0)] = 0
            else:
                signals[breakout_up] = 2
                signals[breakout_down] = 0

        elif ml_predictions is not None:
            # Use ML predictions only
            common_idx = signals.index.intersection(ml_predictions.index)
            signals.loc[common_idx] = ml_predictions.loc[common_idx]

        logger.info(f"Generated {(signals == 2).sum()} buy and {(signals == 0).sum()} sell signals")

        return signals


class PureMLStrategy(TradingStrategy):
    """Pure ML-based strategy."""

    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_predictions: Optional[pd.Series] = None
    ) -> pd.Series:
        """Generate pure ML signals."""
        logger.info("Generating pure ML signals...")

        if ml_predictions is None:
            raise ValueError("ML predictions required for PureMLStrategy")

        signals = pd.Series(1, index=df.index)  # Default: hold

        # Align indices
        common_idx = signals.index.intersection(ml_predictions.index)
        signals.loc[common_idx] = ml_predictions.loc[common_idx]

        logger.info(f"Generated {(signals == 2).sum()} buy and {(signals == 0).sum()} sell signals")

        return signals


class StrategyFactory:
    """Factory for creating trading strategies."""

    @staticmethod
    def create_strategy(strategy_name: str, config: dict) -> TradingStrategy:
        """
        Create a trading strategy.

        Args:
            strategy_name: Name of the strategy
            config: Strategy configuration

        Returns:
            Strategy instance
        """
        strategies = {
            'momentum_ml': MomentumMLStrategy,
            'mean_reversion_ml': MeanReversionMLStrategy,
            'volatility_breakout_ml': VolatilityBreakoutMLStrategy,
            'pure_ml': PureMLStrategy
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        logger.info(f"Creating {strategy_name} strategy")
        return strategies[strategy_name](config)
