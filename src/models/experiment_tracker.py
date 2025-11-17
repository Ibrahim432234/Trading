"""Experiment tracking for ML models."""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger("trading_system")


class ExperimentTracker:
    """Track ML experiments and results."""

    def __init__(self, experiments_dir: str = "data/experiments"):
        """
        Initialize experiment tracker.

        Args:
            experiments_dir: Directory to store experiments
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_file = self.experiments_dir / "experiments.json"
        self.experiments = self._load_experiments()

    def _load_experiments(self) -> list:
        """Load existing experiments."""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        return []

    def _save_experiments(self):
        """Save experiments to file."""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)

    def log_experiment(
        self,
        experiment_name: str,
        model_name: str,
        model_params: Dict,
        model_metrics: Dict,
        backtest_results: Dict,
        config: Dict,
        feature_names: list,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Log a new experiment.

        Args:
            experiment_name: Name of the experiment
            model_name: Name of the model
            model_params: Model parameters
            model_metrics: Model metrics
            backtest_results: Backtest results
            config: Configuration used
            feature_names: List of feature names
            metadata: Additional metadata

        Returns:
            Experiment ID
        """
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Clean non-serializable data
        clean_backtest = {k: v for k, v in backtest_results.items()
                         if k not in ['trades', 'equity_curve', 'equity_dates']}

        clean_metrics = {k: v for k, v in model_metrics.items()
                        if k not in ['classification_report']}

        experiment = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'model_params': model_params,
            'model_metrics': clean_metrics,
            'backtest_results': clean_backtest,
            'config': config,
            'feature_count': len(feature_names),
            'metadata': metadata or {}
        }

        self.experiments.append(experiment)
        self._save_experiments()

        logger.info(f"Logged experiment: {experiment_id}")

        return experiment_id

    def get_experiment(self, experiment_id: str) -> Optional[Dict]:
        """Get experiment by ID."""
        for exp in self.experiments:
            if exp['experiment_id'] == experiment_id:
                return exp
        return None

    def get_experiments_df(self) -> pd.DataFrame:
        """Get all experiments as DataFrame."""
        if not self.experiments:
            return pd.DataFrame()

        # Flatten nested dictionaries for DataFrame
        flat_experiments = []
        for exp in self.experiments:
            flat_exp = {
                'experiment_id': exp['experiment_id'],
                'experiment_name': exp['experiment_name'],
                'timestamp': exp['timestamp'],
                'model_name': exp['model_name'],
                'accuracy': exp['model_metrics'].get('accuracy', 0),
                'f1_score': exp['model_metrics'].get('f1', 0),
                'total_return': exp['backtest_results'].get('total_return', 0),
                'sharpe_ratio': exp['backtest_results'].get('sharpe_ratio', 0),
                'max_drawdown': exp['backtest_results'].get('max_drawdown', 0),
                'win_rate': exp['backtest_results'].get('win_rate', 0),
                'total_trades': exp['backtest_results'].get('total_trades', 0)
            }
            flat_experiments.append(flat_exp)

        return pd.DataFrame(flat_experiments)

    def get_best_experiment(
        self,
        metric: str = 'sharpe_ratio',
        min_trades: int = 10
    ) -> Optional[Dict]:
        """
        Get best experiment based on a metric.

        Args:
            metric: Metric to optimize ('sharpe_ratio', 'total_return', 'win_rate')
            min_trades: Minimum number of trades required

        Returns:
            Best experiment dictionary
        """
        df = self.get_experiments_df()

        if df.empty:
            return None

        # Filter by minimum trades
        df = df[df['total_trades'] >= min_trades]

        if df.empty:
            logger.warning(f"No experiments with >= {min_trades} trades")
            return None

        # Sort by metric
        best_idx = df[metric].idxmax()
        best_id = df.loc[best_idx, 'experiment_id']

        return self.get_experiment(best_id)

    def compare_experiments(
        self,
        experiment_ids: list,
        metrics: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs
            metrics: List of metrics to compare

        Returns:
            Comparison DataFrame
        """
        if metrics is None:
            metrics = ['accuracy', 'f1_score', 'total_return', 'sharpe_ratio', 'max_drawdown']

        df = self.get_experiments_df()
        df = df[df['experiment_id'].isin(experiment_ids)]

        return df[['experiment_id', 'model_name'] + metrics]

    def cleanup_old_experiments(self, keep_last_n: int = 50):
        """
        Remove old experiments, keeping only the last N.

        Args:
            keep_last_n: Number of experiments to keep
        """
        if len(self.experiments) > keep_last_n:
            self.experiments = sorted(
                self.experiments,
                key=lambda x: x['timestamp'],
                reverse=True
            )[:keep_last_n]
            self._save_experiments()
            logger.info(f"Cleaned up experiments, kept {keep_last_n}")

    def generate_leaderboard(self, top_n: int = 10) -> pd.DataFrame:
        """
        Generate leaderboard of top experiments.

        Args:
            top_n: Number of top experiments

        Returns:
            Leaderboard DataFrame
        """
        df = self.get_experiments_df()

        if df.empty:
            return pd.DataFrame()

        # Sort by Sharpe ratio (primary) and total return (secondary)
        df = df.sort_values(['sharpe_ratio', 'total_return'], ascending=False)

        leaderboard = df.head(top_n)[
            ['experiment_id', 'model_name', 'sharpe_ratio', 'total_return',
             'max_drawdown', 'win_rate', 'total_trades']
        ]

        return leaderboard
