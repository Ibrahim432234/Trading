"""Live trading implementation with safety controls."""

import ccxt
import time
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import pandas as pd
import os

from ..features.feature_engineer import FeatureEngineer
from ..data.data_cleaner import DataCleaner

logger = logging.getLogger("trading_system")


class LiveTrader:
    """Live trading bot with paper and real trading modes."""

    def __init__(self, config: dict, model_path: str, mode: str = "paper"):
        """
        Initialize live trader.

        Args:
            config: Trading configuration
            model_path: Path to trained model
            mode: 'paper' or 'live'
        """
        self.config = config
        self.mode = mode
        self.symbol = config['data']['symbol']
        self.timeframe = config['data']['timeframes'][0]

        # Load model
        self.model_data = self._load_model(model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.feature_names = self.model_data['feature_names']

        # Initialize exchange
        self.exchange = self._initialize_exchange()

        # Trading state
        self.position = None
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.emergency_stop = False

        # Risk limits
        self.max_daily_loss = float(os.getenv('MAX_DAILY_LOSS', 0.05))
        self.max_drawdown = float(os.getenv('MAX_DRAWDOWN', 0.20))
        self.max_daily_trades = config['live_trading'].get('max_daily_trades', 10)
        self.position_size = config['backtest'].get('position_size', 0.02)

        # Feature engineer
        self.feature_engineer = FeatureEngineer(config['features'])

        logger.info(f"Live trader initialized in {mode} mode")

    def _load_model(self, model_path: str) -> Dict:
        """Load trained model."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_data = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        return model_data

    def _initialize_exchange(self):
        """Initialize exchange connection."""
        exchange_name = self.config['data']['exchange']

        if self.mode == 'paper':
            # Paper trading - no API keys needed
            logger.info("Initializing paper trading mode")
            exchange_class = getattr(ccxt, exchange_name.lower())
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
        else:
            # Live trading - API keys required
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')

            if not api_key or not api_secret:
                raise ValueError("API credentials not found in environment")

            logger.info("Initializing live trading mode")
            exchange_class = getattr(ccxt, exchange_name.lower())
            exchange = exchange_class({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })

        return exchange

    def get_current_price(self) -> float:
        """Get current market price."""
        ticker = self.exchange.fetch_ticker(self.symbol)
        return ticker['last']

    def fetch_recent_data(self, lookback_periods: int = 200) -> pd.DataFrame:
        """
        Fetch recent market data for feature generation.

        Args:
            lookback_periods: Number of periods to fetch

        Returns:
            DataFrame with OHLCV data
        """
        ohlcv = self.exchange.fetch_ohlcv(
            self.symbol,
            timeframe=self.timeframe,
            limit=lookback_periods
        )

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Clean data
        cleaner = DataCleaner()
        df = cleaner.clean_ohlcv(df, fill_missing=True)

        return df

    def generate_signal(self, df: pd.DataFrame) -> int:
        """
        Generate trading signal.

        Args:
            df: OHLCV DataFrame

        Returns:
            Signal (0=sell, 1=hold, 2=buy)
        """
        # Generate features
        df = self.feature_engineer.create_features(df)

        # Get latest row features
        latest = df.iloc[[-1]][self.feature_names]

        # Handle missing features
        latest = latest.fillna(0)

        # Scale
        latest_scaled = self.scaler.transform(latest)

        # Predict
        signal = self.model.predict(latest_scaled)[0]

        logger.debug(f"Generated signal: {signal}")

        return int(signal)

    def check_risk_limits(self) -> bool:
        """
        Check if risk limits are violated.

        Returns:
            True if trading can continue, False if emergency stop needed
        """
        # Check daily loss
        if abs(self.daily_pnl) > self.max_daily_loss * self.current_equity:
            logger.error(f"Daily loss limit exceeded: {self.daily_pnl:.2f}")
            return False

        # Check max trades
        if self.daily_trades >= self.max_daily_trades:
            logger.warning(f"Max daily trades reached: {self.daily_trades}")
            return False

        # Check drawdown
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
            if drawdown > self.max_drawdown:
                logger.error(f"Max drawdown exceeded: {drawdown:.2%}")
                return False

        return True

    def execute_trade(self, signal: int, current_price: float):
        """
        Execute trade based on signal.

        Args:
            signal: Trading signal
            current_price: Current market price
        """
        if self.emergency_stop:
            logger.warning("Trading stopped due to emergency stop")
            return

        if not self.check_risk_limits():
            logger.error("Risk limits violated. Activating emergency stop.")
            self.emergency_stop = True
            self.close_position(current_price)
            return

        # Buy signal
        if signal == 2 and self.position is None:
            self.open_position('long', current_price)

        # Sell signal
        elif signal == 0 and self.position is not None:
            self.close_position(current_price)

    def open_position(self, direction: str, price: float):
        """
        Open a position.

        Args:
            direction: 'long' or 'short'
            price: Entry price
        """
        if self.mode == 'paper':
            # Simulate position
            size = self.current_equity * self.position_size / price

            self.position = {
                'direction': direction,
                'entry_price': price,
                'size': size,
                'entry_time': datetime.now()
            }

            logger.info(f"📊 Opened {direction} position at ${price:.2f} (size: {size:.4f})")

        else:
            # Real trading
            try:
                balance = self.exchange.fetch_balance()
                base_currency = self.symbol.split('/')[1]
                available = balance[base_currency]['free']

                size = (available * self.position_size) / price

                # Place market order
                order = self.exchange.create_market_buy_order(self.symbol, size)

                self.position = {
                    'direction': direction,
                    'entry_price': price,
                    'size': size,
                    'entry_time': datetime.now(),
                    'order_id': order['id']
                }

                logger.info(f"✅ Executed {direction} order: {size:.4f} @ ${price:.2f}")

            except Exception as e:
                logger.error(f"Failed to execute order: {e}")

        self.daily_trades += 1

    def close_position(self, price: float):
        """
        Close current position.

        Args:
            price: Exit price
        """
        if self.position is None:
            return

        if self.mode == 'paper':
            # Simulate close
            pnl = (price - self.position['entry_price']) * self.position['size']

            logger.info(f"📊 Closed {self.position['direction']} position at ${price:.2f} (PnL: ${pnl:.2f})")

            self.daily_pnl += pnl
            self.current_equity += pnl

            self.position = None

        else:
            # Real trading
            try:
                size = self.position['size']
                order = self.exchange.create_market_sell_order(self.symbol, size)

                pnl = (price - self.position['entry_price']) * size

                logger.info(f"✅ Closed position: {size:.4f} @ ${price:.2f} (PnL: ${pnl:.2f})")

                self.daily_pnl += pnl
                self.current_equity += pnl

                self.position = None

            except Exception as e:
                logger.error(f"Failed to close position: {e}")

    def run(self, check_interval: int = 60):
        """
        Run trading loop.

        Args:
            check_interval: Interval between checks (seconds)
        """
        logger.info(f"Starting live trading in {self.mode} mode...")
        logger.info(f"Symbol: {self.symbol}, Timeframe: {self.timeframe}")
        logger.info(f"Check interval: {check_interval}s")

        # Initialize equity
        if self.mode == 'paper':
            self.current_equity = self.config['backtest']['initial_capital']
        else:
            balance = self.exchange.fetch_balance()
            base_currency = self.symbol.split('/')[1]
            self.current_equity = balance[base_currency]['free']

        self.peak_equity = self.current_equity

        last_day = datetime.now().date()

        try:
            while not self.emergency_stop:
                try:
                    # Reset daily counters
                    current_day = datetime.now().date()
                    if current_day != last_day:
                        logger.info(f"Day changed. Resetting daily counters. PnL: ${self.daily_pnl:.2f}")
                        self.daily_pnl = 0.0
                        self.daily_trades = 0
                        last_day = current_day

                    # Fetch data
                    df = self.fetch_recent_data()

                    # Generate signal
                    signal = self.generate_signal(df)

                    # Get current price
                    current_price = self.get_current_price()

                    # Execute trade
                    self.execute_trade(signal, current_price)

                    # Update peak equity
                    if self.current_equity > self.peak_equity:
                        self.peak_equity = self.current_equity

                    # Log status
                    logger.info(f"Status - Price: ${current_price:.2f}, Signal: {signal}, "
                              f"Position: {self.position is not None}, "
                              f"Equity: ${self.current_equity:.2f}, "
                              f"Daily PnL: ${self.daily_pnl:.2f}")

                    # Wait
                    time.sleep(check_interval)

                except KeyboardInterrupt:
                    logger.info("Received keyboard interrupt. Shutting down...")
                    break

                except Exception as e:
                    logger.error(f"Error in trading loop: {e}", exc_info=True)
                    time.sleep(check_interval)

        finally:
            # Close any open positions
            if self.position is not None:
                logger.info("Closing open position before shutdown...")
                current_price = self.get_current_price()
                self.close_position(current_price)

            logger.info("Trading stopped.")
            logger.info(f"Final equity: ${self.current_equity:.2f}")
            logger.info(f"Total PnL: ${self.current_equity - self.config['backtest']['initial_capital']:.2f}")
