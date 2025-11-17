#!/usr/bin/env python3
"""
Training script for ML trading system.

Usage:
    python train.py --experiment-name my_experiment
    python train.py --experiment-name test --no-fetch  # Use existing data
    python train.py --multiple-models  # Train all configured models
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.utils.logger import setup_logger
from src.models.automl_loop import AutoMLLoop


def main():
    parser = argparse.ArgumentParser(description='Train ML trading model')
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='experiment',
        help='Name of the experiment'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--no-fetch',
        action='store_true',
        help='Use existing data instead of fetching new'
    )
    parser.add_argument(
        '--multiple-models',
        action='store_true',
        help='Train multiple models and compare'
    )

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger(log_file='data/logs/training.log')
    logger.info("="*80)
    logger.info("ML TRADING SYSTEM - TRAINING")
    logger.info("="*80)

    try:
        # Initialize AutoML loop
        automl = AutoMLLoop(config_path=args.config)

        # Run pipeline
        if args.multiple_models:
            logger.info("Training multiple models...")
            results = automl.run_multiple_models(args.experiment_name)

            # Display summary
            logger.info("\n" + "="*80)
            logger.info("TRAINING COMPLETE - SUMMARY")
            logger.info("="*80)

            for result in results:
                if result['success']:
                    logger.info(f"\n{result['experiment_id']}:")
                    logger.info(f"  Model Accuracy: {result['model_metrics']['accuracy']:.4f}")
                    logger.info(f"  Sharpe Ratio: {result['backtest_results']['sharpe_ratio']:.2f}")
                    logger.info(f"  Total Return: {result['backtest_results']['total_return']*100:.2f}%")
        else:
            logger.info(f"Training experiment: {args.experiment_name}")
            result = automl.run_full_pipeline(
                experiment_name=args.experiment_name,
                fetch_new_data=not args.no_fetch
            )

            if result['success']:
                logger.info("\n" + "="*80)
                logger.info("TRAINING COMPLETE")
                logger.info("="*80)
                logger.info(f"Experiment ID: {result['experiment_id']}")
                logger.info(f"Model Accuracy: {result['model_metrics']['accuracy']:.4f}")
                logger.info(f"Sharpe Ratio: {result['backtest_results']['sharpe_ratio']:.2f}")
                logger.info(f"Total Return: {result['backtest_results']['total_return']*100:.2f}%")
                logger.info(f"Max Drawdown: {result['backtest_results']['max_drawdown']*100:.2f}%")
            else:
                logger.error(f"Training failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
