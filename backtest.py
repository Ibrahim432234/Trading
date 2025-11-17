#!/usr/bin/env python3
"""
Backtesting script for ML trading system.

Usage:
    python backtest.py --model data/models/xgboost_20231201_120000.pkl
    python backtest.py --model data/models/xgboost_20231201_120000.pkl --strategy momentum_ml
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.data.data_fetcher import load_data
from src.features.feature_engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.backtest.backtester import Backtester
from src.backtest.strategies import StrategyFactory
from src.evaluation.visualizer import Visualizer
from src.evaluation.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description='Backtest ML trading model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to data file (if not specified, will fetch)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='pure_ml',
        choices=['momentum_ml', 'mean_reversion_ml', 'volatility_breakout_ml', 'pure_ml'],
        help='Trading strategy'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/backtest',
        help='Output directory for reports'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(log_file='data/logs/backtest.log')
    logger.info("="*80)
    logger.info("ML TRADING SYSTEM - BACKTEST")
    logger.info("="*80)

    try:
        # Load config
        config = load_config(args.config)

        # Load model
        logger.info(f"Loading model from {args.model}")
        model_data = ModelTrainer.load_model(args.model)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']

        # Load data
        if args.data:
            logger.info(f"Loading data from {args.data}")
            df = load_data(args.data)
        else:
            logger.info("Fetching data...")
            from src.data.data_fetcher import DataFetcher
            from src.data.data_cleaner import DataCleaner

            fetcher = DataFetcher(config['data']['exchange'])
            df = fetcher.fetch_ohlcv(
                symbol=config['data']['symbol'],
                timeframe=config['data']['timeframes'][0],
                start_date=config['data'].get('start_date')
            )

            cleaner = DataCleaner()
            df = cleaner.clean_ohlcv(df)

        # Generate features
        logger.info("Generating features...")
        engineer = FeatureEngineer(config['features'])
        df = engineer.create_features(df)

        # Prepare features for prediction
        X = df[feature_names].fillna(0)
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            index=X.index,
            columns=X.columns
        )

        # Generate predictions
        logger.info("Generating predictions...")
        predictions = pd.Series(
            model.predict(X_scaled),
            index=X_scaled.index
        )

        # Create strategy
        logger.info(f"Using strategy: {args.strategy}")
        strategy = StrategyFactory.create_strategy(args.strategy, config['backtest'])

        # Generate signals
        signals = strategy.generate_signals(df, predictions)

        # Run backtest
        logger.info("Running backtest...")
        backtester = Backtester(config['backtest'])
        results = backtester.run(df, signals)

        # Display results
        logger.info("\n" + "="*80)
        logger.info("BACKTEST RESULTS")
        logger.info("="*80)
        logger.info(f"Initial Capital:  ${results['initial_capital']:,.2f}")
        logger.info(f"Final Equity:     ${results['final_equity']:,.2f}")
        logger.info(f"Total Return:     {results['total_return']*100:.2f}%")
        logger.info(f"CAGR:             {results['cagr']*100:.2f}%")
        logger.info(f"Sharpe Ratio:     {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown:     {results['max_drawdown']*100:.2f}%")
        logger.info(f"Total Trades:     {results['total_trades']}")
        logger.info(f"Win Rate:         {results['win_rate']*100:.2f}%")

        # Generate reports
        logger.info(f"\nGenerating reports in {args.output}...")

        # Get feature importance
        trainer = ModelTrainer(config['models'])
        importance_df = trainer.get_feature_importance(model, feature_names)

        # Visualizations
        Visualizer.plot_equity_curve(
            results['equity_curve'],
            results['equity_dates'],
            save_path=f"{args.output}/equity_curve.png"
        )

        Visualizer.plot_drawdown(
            results['equity_curve'],
            results['equity_dates'],
            save_path=f"{args.output}/drawdown.png"
        )

        if results.get('trades'):
            Visualizer.plot_returns_distribution(
                results['trades'],
                save_path=f"{args.output}/returns_distribution.png"
            )

            # Trade log
            trade_log = backtester.get_trade_log()
            if not trade_log.empty:
                trade_log.to_csv(f"{args.output}/trade_log.csv")
                logger.info(f"Trade log saved to {args.output}/trade_log.csv")

        # Text report
        ReportGenerator.generate_text_report(
            results,
            model_data.get('metadata', {}),
            f"{args.output}/report.txt"
        )

        logger.info("\n" + "="*80)
        logger.info("BACKTEST COMPLETE")
        logger.info("="*80)
        logger.info(f"Reports saved to: {args.output}/")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
