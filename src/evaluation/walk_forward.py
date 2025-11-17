"""Walk-forward analysis implementation."""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import Dict, List
from pathlib import Path

from ..models.trainer import ModelTrainer
from ..backtest.backtester import Backtester
from ..backtest.strategies import StrategyFactory
from ..features.feature_engineer import FeatureEngineer

logger = logging.getLogger("trading_system")


class WalkForwardAnalysis:
    """Perform walk-forward analysis for strategy validation."""

    def __init__(self, config: dict):
        """
        Initialize walk-forward analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.wf_config = config['evaluation']['walk_forward']

    def run(
        self,
        df: pd.DataFrame,
        model_name: str = 'xgboost',
        strategy_name: str = 'pure_ml'
    ) -> Dict:
        """
        Run walk-forward analysis.

        Args:
            df: DataFrame with OHLCV data
            model_name: Name of ML model to use
            strategy_name: Name of trading strategy

        Returns:
            Dictionary with walk-forward results
        """
        logger.info("Starting walk-forward analysis...")

        train_period = self.wf_config['train_period']  # days
        test_period = self.wf_config['test_period']    # days
        step = self.wf_config['step']                   # days

        # Initialize feature engineer
        engineer = FeatureEngineer(self.config['features'])

        # Generate features once
        df_features = engineer.create_features(df)
        df_labeled = engineer.create_labels(
            df_features,
            label_type=self.config['features']['labels']['type'],
            horizons=self.config['features']['labels']['horizon'],
            threshold=self.config['features']['labels']['threshold']
        )

        # Prepare ML data
        target_col = f"target_{self.config['features']['labels']['horizon'][0]}"
        X, y, feature_names = engineer.prepare_ml_data(df_labeled, target_col=target_col)

        # Get date range
        start_date = df.index.min()
        end_date = df.index.max()

        # Calculate windows
        windows = []
        current_start = start_date

        while current_start + timedelta(days=train_period + test_period) <= end_date:
            train_end = current_start + timedelta(days=train_period)
            test_end = train_end + timedelta(days=test_period)

            windows.append({
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })

            current_start += timedelta(days=step)

        logger.info(f"Created {len(windows)} walk-forward windows")

        # Run walk-forward
        window_results = []

        for i, window in enumerate(windows):
            logger.info(f"\nWindow {i+1}/{len(windows)}")
            logger.info(f"Train: {window['train_start']} to {window['train_end']}")
            logger.info(f"Test:  {window['test_start']} to {window['test_end']}")

            try:
                result = self._run_window(
                    X, y, df_labeled, feature_names,
                    window, model_name, strategy_name
                )
                window_results.append(result)

            except Exception as e:
                logger.error(f"Window {i+1} failed: {e}")
                continue

        # Aggregate results
        aggregated = self._aggregate_results(window_results)

        logger.info("\n" + "="*80)
        logger.info("WALK-FORWARD ANALYSIS COMPLETE")
        logger.info("="*80)
        logger.info(f"Windows:        {len(window_results)}")
        logger.info(f"Avg Sharpe:     {aggregated['avg_sharpe']:.2f}")
        logger.info(f"Avg Return:     {aggregated['avg_return']*100:.2f}%")
        logger.info(f"Avg Max DD:     {aggregated['avg_max_dd']*100:.2f}%")
        logger.info(f"Win Rate:       {aggregated['win_rate']*100:.2f}%")

        return {
            'windows': window_results,
            'aggregated': aggregated
        }

    def _run_window(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        df_labeled: pd.DataFrame,
        feature_names: list,
        window: dict,
        model_name: str,
        strategy_name: str
    ) -> Dict:
        """Run single walk-forward window."""

        # Split data by date
        X_train = X[(X.index >= window['train_start']) & (X.index < window['train_end'])]
        y_train = y[(y.index >= window['train_start']) & (y.index < window['train_end'])]

        X_test = X[(X.index >= window['test_start']) & (X.index < window['test_end'])]
        y_test = y[(y.index >= window['test_start']) & (y.index < window['test_end'])]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError("Empty train or test set")

        # Train model
        trainer = ModelTrainer(self.config['models'])
        trainer.feature_names = feature_names

        # Scale
        X_train_scaled = pd.DataFrame(
            trainer.scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )

        X_test_scaled = pd.DataFrame(
            trainer.scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )

        # Train
        model = trainer.train_model(model_name, X_train_scaled, y_train)

        # Predict
        predictions = pd.Series(
            model.predict(X_test_scaled),
            index=X_test.index
        )

        # Backtest
        strategy = StrategyFactory.create_strategy(strategy_name, self.config['backtest'])
        signals = strategy.generate_signals(
            df_labeled.loc[window['test_start']:window['test_end']],
            predictions
        )

        backtester = Backtester(self.config['backtest'])
        results = backtester.run(
            df_labeled.loc[window['test_start']:window['test_end']],
            signals
        )

        return {
            'window': window,
            'sharpe_ratio': results['sharpe_ratio'],
            'total_return': results['total_return'],
            'max_drawdown': results['max_drawdown'],
            'win_rate': results['win_rate'],
            'total_trades': results['total_trades']
        }

    def _aggregate_results(self, window_results: List[Dict]) -> Dict:
        """Aggregate walk-forward results."""

        if not window_results:
            return {}

        sharpe_ratios = [w['sharpe_ratio'] for w in window_results]
        returns = [w['total_return'] for w in window_results]
        max_dds = [w['max_drawdown'] for w in window_results]
        win_rates = [w['win_rate'] for w in window_results]

        # Count profitable windows
        profitable_windows = sum(1 for r in returns if r > 0)

        return {
            'avg_sharpe': np.mean(sharpe_ratios),
            'std_sharpe': np.std(sharpe_ratios),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_max_dd': np.mean(max_dds),
            'win_rate': profitable_windows / len(window_results),
            'total_windows': len(window_results),
            'profitable_windows': profitable_windows
        }

    def plot_results(self, results: Dict, save_path: str = None):
        """Plot walk-forward results."""
        import matplotlib.pyplot as plt

        windows = results['windows']

        if not windows:
            logger.warning("No windows to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Sharpe Ratio
        sharpe = [w['sharpe_ratio'] for w in windows]
        axes[0, 0].plot(range(len(sharpe)), sharpe, marker='o')
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_title('Sharpe Ratio per Window')
        axes[0, 0].set_xlabel('Window')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].grid(True, alpha=0.3)

        # Returns
        returns = [w['total_return'] * 100 for w in windows]
        colors = ['g' if r > 0 else 'r' for r in returns]
        axes[0, 1].bar(range(len(returns)), returns, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-')
        axes[0, 1].set_title('Returns per Window')
        axes[0, 1].set_xlabel('Window')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].grid(True, alpha=0.3)

        # Max Drawdown
        max_dds = [w['max_drawdown'] * 100 for w in windows]
        axes[1, 0].bar(range(len(max_dds)), max_dds, color='red', alpha=0.7)
        axes[1, 0].set_title('Max Drawdown per Window')
        axes[1, 0].set_xlabel('Window')
        axes[1, 0].set_ylabel('Max DD (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Win Rate
        win_rates = [w['win_rate'] * 100 for w in windows]
        axes[1, 1].plot(range(len(win_rates)), win_rates, marker='o', color='orange')
        axes[1, 1].axhline(y=50, color='gray', linestyle='--')
        axes[1, 1].set_title('Win Rate per Window')
        axes[1, 1].set_xlabel('Window')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Walk-forward plot saved to {save_path}")

        plt.show()
