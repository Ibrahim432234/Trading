#!/usr/bin/env python3
"""Optimize strategy parameters."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from itertools import product

from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.data.data_fetcher import load_data
from src.data.data_cleaner import DataCleaner
from src.features.feature_engineer import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.backtest.backtester import Backtester
from src.backtest.strategies import StrategyFactory


def main():
    parser = argparse.ArgumentParser(description='Optimize strategy parameters')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--strategy', type=str, default='pure_ml')
    parser.add_argument('--config', type=str, default='config/config.yaml')

    args = parser.parse_args()

    logger = setup_logger(log_file='data/logs/optimization.log')
    logger.info("="*80)
    logger.info("STRATEGY PARAMETER OPTIMIZATION")
    logger.info("="*80)

    try:
        # Load config
        config = load_config(args.config)

        # Load data
        df = load_data(args.data)
        cleaner = DataCleaner()
        df = cleaner.clean_ohlcv(df)

        # Load model
        model_data = ModelTrainer.load_model(args.model)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']

        # Generate features
        engineer = FeatureEngineer(config['features'])
        df_features = engineer.create_features(df)

        # Prepare features
        X = df_features[feature_names].fillna(0)
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            index=X.index,
            columns=X.columns
        )

        # Generate predictions
        predictions = pd.Series(model.predict(X_scaled), index=X_scaled.index)

        # Parameter grid
        position_sizes = [0.01, 0.02, 0.05, 0.10]
        stop_losses = [0.01, 0.02, 0.03, 0.05]
        take_profits = [0.02, 0.04, 0.06, 0.08]

        logger.info(f"\nParameter grid:")
        logger.info(f"Position sizes: {position_sizes}")
        logger.info(f"Stop losses:    {stop_losses}")
        logger.info(f"Take profits:   {take_profits}")
        logger.info(f"Total combinations: {len(position_sizes) * len(stop_losses) * len(take_profits)}")

        results = []

        for pos_size, stop_loss, take_profit in product(position_sizes, stop_losses, take_profits):
            # Update config
            test_config = config.copy()
            test_config['backtest']['position_size'] = pos_size
            test_config['backtest']['risk_management']['stop_loss'] = stop_loss
            test_config['backtest']['risk_management']['take_profit'] = take_profit

            # Run backtest
            strategy = StrategyFactory.create_strategy(args.strategy, test_config['backtest'])
            signals = strategy.generate_signals(df_features, predictions)

            backtester = Backtester(test_config['backtest'])
            bt_results = backtester.run(df_features, signals)

            results.append({
                'position_size': pos_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sharpe_ratio': bt_results['sharpe_ratio'],
                'total_return': bt_results['total_return'],
                'max_drawdown': bt_results['max_drawdown'],
                'win_rate': bt_results['win_rate'],
                'total_trades': bt_results['total_trades']
            })

            logger.debug(f"Tested: pos={pos_size}, sl={stop_loss}, tp={take_profit} "
                        f"-> Sharpe={bt_results['sharpe_ratio']:.2f}")

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Find best parameters
        best_idx = results_df['sharpe_ratio'].idxmax()
        best = results_df.loc[best_idx]

        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("="*80)
        logger.info(f"Best Parameters:")
        logger.info(f"  Position Size: {best['position_size']:.2%}")
        logger.info(f"  Stop Loss:     {best['stop_loss']:.2%}")
        logger.info(f"  Take Profit:   {best['take_profit']:.2%}")
        logger.info(f"\nPerformance:")
        logger.info(f"  Sharpe Ratio:  {best['sharpe_ratio']:.2f}")
        logger.info(f"  Total Return:  {best['total_return']*100:.2f}%")
        logger.info(f"  Max Drawdown:  {best['max_drawdown']*100:.2f}%")
        logger.info(f"  Win Rate:      {best['win_rate']*100:.2f}%")

        # Save results
        results_df.to_csv('reports/optimization_results.csv', index=False)
        logger.info(f"\nFull results saved to reports/optimization_results.csv")

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
