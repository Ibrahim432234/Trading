#!/usr/bin/env python3
"""Run walk-forward analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.data.data_fetcher import DataFetcher, load_data
from src.data.data_cleaner import DataCleaner
from src.evaluation.walk_forward import WalkForwardAnalysis


def main():
    parser = argparse.ArgumentParser(description='Run walk-forward analysis')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--model', type=str, default='xgboost')
    parser.add_argument('--strategy', type=str, default='pure_ml')
    parser.add_argument('--output', type=str, default='reports/walk_forward')

    args = parser.parse_args()

    logger = setup_logger(log_file='data/logs/walk_forward.log')
    logger.info("="*80)
    logger.info("WALK-FORWARD ANALYSIS")
    logger.info("="*80)

    try:
        # Load config
        config = load_config(args.config)

        # Load or fetch data
        if args.data:
            df = load_data(args.data)
        else:
            fetcher = DataFetcher(config['data']['exchange'])
            df = fetcher.fetch_ohlcv(
                symbol=config['data']['symbol'],
                timeframe=config['data']['timeframes'][0],
                start_date=config['data'].get('start_date')
            )

        # Clean data
        cleaner = DataCleaner()
        df = cleaner.clean_ohlcv(df)

        # Run walk-forward
        wf = WalkForwardAnalysis(config)
        results = wf.run(df, model_name=args.model, strategy_name=args.strategy)

        # Plot results
        wf.plot_results(results, save_path=f"{args.output}/walk_forward.png")

        # Save detailed results
        import json
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / 'walk_forward_results.json', 'w') as f:
            json.dump({
                'aggregated': results['aggregated'],
                'num_windows': len(results['windows'])
            }, f, indent=2, default=str)

        logger.info(f"\nResults saved to {args.output}/")

    except Exception as e:
        logger.error(f"Walk-forward failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
