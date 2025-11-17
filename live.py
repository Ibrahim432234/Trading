#!/usr/bin/env python3
"""
Live trading script for ML trading system.

Usage:
    # Paper trading
    python live.py --model data/models/xgboost_20231201_120000.pkl --mode paper

    # Live trading (requires confirmation)
    python live.py --model data/models/xgboost_20231201_120000.pkl --mode live --confirm
"""

import argparse
import sys
from pathlib import Path
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.live_trading.trader import LiveTrader


def main():
    parser = argparse.ArgumentParser(description='Run live trading')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='paper',
        choices=['paper', 'live'],
        help='Trading mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Check interval in seconds'
    )
    parser.add_argument(
        '--confirm',
        action='store_true',
        help='Confirm live trading (required for live mode)'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(log_file='data/logs/live_trading.log')
    logger.info("="*80)
    logger.info("ML TRADING SYSTEM - LIVE TRADING")
    logger.info("="*80)

    # Safety check for live mode
    if args.mode == 'live':
        if not args.confirm:
            logger.error("Live trading requires --confirm flag")
            logger.error("Please read the documentation and ensure you understand the risks")
            sys.exit(1)

        # Check API credentials
        if not os.getenv('BINANCE_API_KEY') or not os.getenv('BINANCE_API_SECRET'):
            logger.error("API credentials not found in environment")
            logger.error("Please set BINANCE_API_KEY and BINANCE_API_SECRET")
            sys.exit(1)

        # Final confirmation
        logger.warning("⚠️  LIVE TRADING MODE ENABLED ⚠️")
        logger.warning("This will place REAL orders with REAL money!")
        response = input("\nType 'I UNDERSTAND THE RISKS' to continue: ")

        if response != "I UNDERSTAND THE RISKS":
            logger.info("Live trading cancelled")
            sys.exit(0)

    try:
        # Load config
        config = load_config(args.config)

        # Override mode from environment
        mode = os.getenv('TRADING_MODE', args.mode)

        # Initialize trader
        logger.info(f"Initializing trader in {mode} mode...")
        trader = LiveTrader(
            config=config,
            model_path=args.model,
            mode=mode
        )

        # Display configuration
        logger.info("\n" + "="*80)
        logger.info("TRADING CONFIGURATION")
        logger.info("="*80)
        logger.info(f"Mode:             {mode.upper()}")
        logger.info(f"Symbol:           {trader.symbol}")
        logger.info(f"Timeframe:        {trader.timeframe}")
        logger.info(f"Position Size:    {trader.position_size*100:.1f}%")
        logger.info(f"Max Daily Loss:   {trader.max_daily_loss*100:.1f}%")
        logger.info(f"Max Drawdown:     {trader.max_drawdown*100:.1f}%")
        logger.info(f"Max Daily Trades: {trader.max_daily_trades}")
        logger.info(f"Check Interval:   {args.interval}s")
        logger.info("="*80 + "\n")

        if mode == 'paper':
            logger.info("📊 Starting paper trading (simulation mode)")
        else:
            logger.warning("💰 Starting LIVE trading with real money!")

        # Run trader
        trader.run(check_interval=args.interval)

    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt. Shutting down gracefully...")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Live trading failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
