#!/usr/bin/env python3
"""Compare different models and generate report."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from src.utils.logger import setup_logger
from src.models.experiment_tracker import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description='Compare model experiments')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top models')
    parser.add_argument('--metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'total_return', 'win_rate'])
    parser.add_argument('--output', type=str, default='reports/model_comparison.csv')

    args = parser.parse_args()

    logger = setup_logger()
    logger.info("="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)

    try:
        tracker = ExperimentTracker()

        # Get leaderboard
        leaderboard = tracker.generate_leaderboard(top_n=args.top_n)

        if leaderboard.empty:
            logger.warning("No experiments found")
            return

        # Display
        print("\n" + "="*100)
        print(f"TOP {args.top_n} MODELS (sorted by {args.metric})")
        print("="*100)
        print(leaderboard.to_string())
        print("="*100)

        # Save to CSV
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        leaderboard.to_csv(output_path)

        logger.info(f"\nLeaderboard saved to {args.output}")

        # Summary statistics
        logger.info("\nSUMMARY STATISTICS:")
        logger.info(f"Total experiments: {len(tracker.experiments)}")
        logger.info(f"Best Sharpe:       {leaderboard['sharpe_ratio'].max():.2f}")
        logger.info(f"Best Return:       {leaderboard['total_return'].max()*100:.2f}%")
        logger.info(f"Avg Sharpe:        {leaderboard['sharpe_ratio'].mean():.2f}")

    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
