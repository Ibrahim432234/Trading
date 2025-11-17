"""Data fetching module for cryptocurrency and financial data."""

import ccxt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List
import time
from pathlib import Path
import logging

logger = logging.getLogger("trading_system")


class DataFetcher:
    """Fetch OHLCV data from various sources."""

    def __init__(self, exchange_name: str = "binance"):
        """
        Initialize data fetcher.

        Args:
            exchange_name: Exchange name (e.g., 'binance', 'coinbase')
        """
        self.exchange_name = exchange_name
        self.exchange = None

        if exchange_name.lower() in ['binance', 'coinbase', 'kraken', 'ftx']:
            try:
                exchange_class = getattr(ccxt, exchange_name.lower())
                self.exchange = exchange_class({
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                logger.info(f"Initialized {exchange_name} exchange")
            except Exception as e:
                logger.error(f"Failed to initialize exchange: {e}")
                raise

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data.

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '1h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Number of candles per request

        Returns:
            DataFrame with OHLCV data
        """
        if self.exchange:
            return self._fetch_ccxt(symbol, timeframe, start_date, end_date, limit)
        else:
            # Fallback to yfinance for stocks
            return self._fetch_yfinance(symbol, start_date, end_date, timeframe)

    def _fetch_ccxt(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str],
        limit: int
    ) -> pd.DataFrame:
        """Fetch data using CCXT."""
        logger.info(f"Fetching {symbol} {timeframe} from {self.exchange_name}")

        # Parse dates
        if start_date:
            since = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            since = int((datetime.now() - timedelta(days=365)).timestamp() * 1000)

        if end_date:
            until = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        else:
            until = int(datetime.now().timestamp() * 1000)

        # Fetch data in chunks
        all_ohlcv = []
        current_since = since

        while current_since < until:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=limit
                )

                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1

                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)

                logger.debug(f"Fetched {len(ohlcv)} candles, total: {len(all_ohlcv)}")

            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                break

        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Filter by end date
        if end_date:
            df = df[df.index <= end_date]

        logger.info(f"Fetched {len(df)} candles from {df.index.min()} to {df.index.max()}")

        return df

    def _fetch_yfinance(
        self,
        symbol: str,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str
    ) -> pd.DataFrame:
        """Fetch data using yfinance (for stocks)."""
        logger.info(f"Fetching {symbol} from Yahoo Finance")

        # Map timeframe to yfinance interval
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '1h', '1d': '1d'
        }
        yf_interval = interval_map.get(interval, '1h')

        start = start_date if start_date else (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end = end_date if end_date else datetime.now().strftime("%Y-%m-%d")

        # Download data
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=yf_interval,
            progress=False
        )

        # Standardize column names
        df.columns = [col.lower() for col in df.columns]
        df.index.name = 'timestamp'

        logger.info(f"Fetched {len(df)} candles from Yahoo Finance")

        return df

    def fetch_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> dict:
        """
        Fetch data for multiple timeframes.

        Args:
            symbol: Trading pair
            timeframes: List of timeframes
            start_date: Start date
            end_date: End date

        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        data = {}
        for tf in timeframes:
            try:
                df = self.fetch_ohlcv(symbol, tf, start_date, end_date)
                data[tf] = df
                logger.info(f"Fetched {len(df)} candles for {tf}")
            except Exception as e:
                logger.error(f"Failed to fetch {tf}: {e}")

        return data


def save_data(df: pd.DataFrame, output_path: str, format: str = "parquet") -> None:
    """
    Save DataFrame to file.

    Args:
        df: DataFrame to save
        output_path: Output file path
        format: File format ('csv' or 'parquet')
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if format == "parquet":
        df.to_parquet(output_file)
    else:
        df.to_csv(output_file)

    logger.info(f"Saved data to {output_path}")


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load DataFrame from file.

    Args:
        input_path: Input file path

    Returns:
        Loaded DataFrame
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Data file not found: {input_path}")

    if input_path.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file, index_col=0, parse_dates=True)

    logger.info(f"Loaded {len(df)} rows from {input_path}")

    return df
