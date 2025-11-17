"""AutoML loop for automatic model training and evaluation."""

import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

from ..data.data_fetcher import DataFetcher, load_data, save_data
from ..data.data_cleaner import DataCleaner
from ..features.feature_engineer import FeatureEngineer
from ..models.trainer import ModelTrainer
from ..models.experiment_tracker import ExperimentTracker
from ..backtest.backtester import Backtester
from ..backtest.strategies import StrategyFactory
from ..evaluation.visualizer import Visualizer
from ..evaluation.report_generator import ReportGenerator
from ..utils.config_loader import load_config

logger = logging.getLogger("trading_system")


class AutoMLLoop:
    """Automatic ML training and evaluation loop."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize AutoML loop.

        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.tracker = ExperimentTracker()

    def run_full_pipeline(
        self,
        experiment_name: str,
        fetch_new_data: bool = True
    ) -> Dict:
        """
        Run full ML pipeline.

        Args:
            experiment_name: Name of the experiment
            fetch_new_data: Whether to fetch new data

        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Starting AutoML pipeline: {experiment_name}")
        start_time = datetime.now()

        try:
            # 1. Data Pipeline
            df = self._data_pipeline(fetch_new_data)

            # 2. Feature Engineering
            X, y, feature_names = self._feature_engineering(df)

            # 3. Model Training
            model, model_metrics, best_params = self._model_training(X, y, feature_names)

            # 4. Backtesting
            backtest_results = self._backtesting(df, model, X, y)

            # 5. Evaluation & Reporting
            self._evaluation_and_reporting(
                backtest_results,
                model_metrics,
                model,
                feature_names,
                experiment_name
            )

            # 6. Log Experiment
            experiment_id = self.tracker.log_experiment(
                experiment_name=experiment_name,
                model_name=self.config['models']['algorithms'][0],
                model_params=best_params,
                model_metrics=model_metrics,
                backtest_results=backtest_results,
                config=self.config,
                feature_names=feature_names,
                metadata={
                    'duration_seconds': (datetime.now() - start_time).total_seconds()
                }
            )

            logger.info(f"Pipeline completed: {experiment_id}")
            logger.info(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
            logger.info(f"Total Return: {backtest_results['total_return']*100:.2f}%")

            # Check if model qualifies for deployment
            self._check_deployment_criteria(experiment_id, backtest_results)

            return {
                'experiment_id': experiment_id,
                'model_metrics': model_metrics,
                'backtest_results': backtest_results,
                'success': True
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                'experiment_id': None,
                'error': str(e),
                'success': False
            }

    def _data_pipeline(self, fetch_new: bool) -> pd.DataFrame:
        """Data fetching and cleaning pipeline."""
        logger.info("Running data pipeline...")

        data_config = self.config['data']
        symbol = data_config['symbol']
        timeframe = data_config['timeframes'][0]  # Use first timeframe

        data_file = Path(f"data/raw/{symbol.replace('/', '_')}_{timeframe}.parquet")

        if fetch_new or not data_file.exists():
            # Fetch new data
            fetcher = DataFetcher(data_config['exchange'])
            df = fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=data_config.get('start_date'),
                end_date=data_config.get('end_date')
            )

            # Save raw data
            save_data(df, str(data_file), format='parquet')
        else:
            # Load existing data
            df = load_data(str(data_file))

        # Clean data
        cleaner = DataCleaner()
        df = cleaner.clean_ohlcv(df)

        logger.info(f"Data pipeline complete. Shape: {df.shape}")

        return df

    def _feature_engineering(self, df: pd.DataFrame):
        """Feature engineering pipeline."""
        logger.info("Running feature engineering...")

        engineer = FeatureEngineer(self.config['features'])

        # Create features
        df = engineer.create_features(df)

        # Create labels
        label_config = self.config['features']['labels']
        df = engineer.create_labels(
            df,
            label_type=label_config['type'],
            horizons=label_config['horizon'],
            threshold=label_config['threshold']
        )

        # Prepare ML data
        target_col = f"target_{label_config['horizon'][0]}"
        X, y, feature_names = engineer.prepare_ml_data(df, target_col=target_col)

        logger.info(f"Feature engineering complete. Features: {len(feature_names)}")

        return X, y, feature_names

    def _model_training(self, X, y, feature_names):
        """Model training pipeline."""
        logger.info("Running model training...")

        trainer = ModelTrainer(self.config['models'])
        trainer.feature_names = feature_names

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(
            X, y,
            test_size=self.config['models']['test_size'],
            val_size=self.config['models']['validation_size']
        )

        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = trainer.scale_features(
            X_train, X_val, X_test
        )

        # Get model name
        model_name = self.config['models']['algorithms'][0]

        # Hyperparameter tuning
        best_params = {}
        if self.config['models'].get('hyperparameter_tuning', {}).get('enabled', False):
            method = self.config['models']['hyperparameter_tuning'].get('method', 'optuna')

            if method == 'optuna':
                best_params = trainer.tune_hyperparameters_optuna(
                    model_name,
                    X_train_scaled, y_train,
                    X_val_scaled, y_val,
                    n_trials=self.config['models']['hyperparameter_tuning'].get('n_trials', 50),
                    timeout=self.config['models']['hyperparameter_tuning'].get('timeout', 3600)
                )
            else:
                best_params = trainer.tune_hyperparameters_grid(
                    model_name,
                    X_train_scaled, y_train
                )

        # Train final model
        model = trainer.train_model(
            model_name,
            X_train_scaled,
            y_train,
            params=best_params if best_params else None
        )

        # Evaluate
        model_metrics = trainer.evaluate(model, X_test_scaled, y_test)

        # Save model
        model_path = f"data/models/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        trainer.save_model(model, model_path, metadata=model_metrics)

        logger.info(f"Model training complete. Accuracy: {model_metrics['accuracy']:.4f}")

        return model, model_metrics, best_params

    def _backtesting(self, df, model, X, y):
        """Backtesting pipeline."""
        logger.info("Running backtest...")

        # Generate predictions
        predictions = pd.Series(
            model.predict(X),
            index=X.index
        )

        # Create strategy
        strategy_name = self.config['backtest']['strategies'][0]
        strategy = StrategyFactory.create_strategy(
            strategy_name,
            self.config['backtest']
        )

        # Generate signals
        signals = strategy.generate_signals(df, predictions)

        # Run backtest
        backtester = Backtester(self.config['backtest'])
        results = backtester.run(df, signals)

        logger.info(f"Backtest complete. Sharpe: {results['sharpe_ratio']:.2f}")

        return results

    def _evaluation_and_reporting(
        self,
        backtest_results,
        model_metrics,
        model,
        feature_names,
        experiment_name
    ):
        """Evaluation and reporting pipeline."""
        logger.info("Generating evaluation reports...")

        output_dir = f"reports/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Get feature importance
        trainer = ModelTrainer(self.config['models'])
        importance_df = trainer.get_feature_importance(model, feature_names)

        # Generate visualizations
        Visualizer.create_full_report(
            backtest_results,
            model_metrics,
            importance_df,
            output_dir
        )

        # Generate text reports
        ReportGenerator.generate_full_report(
            backtest_results,
            model_metrics,
            self.config,
            output_dir
        )

        logger.info(f"Reports saved to {output_dir}")

    def _check_deployment_criteria(self, experiment_id: str, results: Dict):
        """Check if model meets deployment criteria."""
        automl_config = self.config.get('automl', {})

        min_sharpe = automl_config.get('min_sharpe', 1.0)
        max_drawdown = automl_config.get('max_drawdown', 0.20)

        if (results['sharpe_ratio'] >= min_sharpe and
            results['max_drawdown'] <= max_drawdown):

            logger.info(f"✓ Model {experiment_id} qualifies for deployment!")
            logger.info(f"  Sharpe: {results['sharpe_ratio']:.2f} >= {min_sharpe}")
            logger.info(f"  MaxDD: {results['max_drawdown']:.2%} <= {max_drawdown:.2%}")

            # Auto-deploy if enabled
            if automl_config.get('auto_deploy', False):
                logger.info("Auto-deployment enabled, but not implemented yet.")
        else:
            logger.warning(f"✗ Model {experiment_id} does not meet deployment criteria")

    def run_walk_forward_analysis(self, experiment_name: str):
        """Run walk-forward analysis."""
        logger.info("Running walk-forward analysis...")

        wf_config = self.config['evaluation']['walk_forward']

        if not wf_config.get('enabled', False):
            logger.warning("Walk-forward analysis not enabled in config")
            return

        # Implementation would iterate through time windows
        # For now, just log that it's not implemented
        logger.warning("Walk-forward analysis implementation pending")

    def run_multiple_models(self, experiment_name: str):
        """Train and compare multiple models."""
        logger.info("Training multiple models...")

        algorithms = self.config['models']['algorithms']
        results = []

        for algo in algorithms:
            # Temporarily set single algorithm
            original_algos = self.config['models']['algorithms']
            self.config['models']['algorithms'] = [algo]

            result = self.run_full_pipeline(f"{experiment_name}_{algo}")
            results.append(result)

            # Restore original
            self.config['models']['algorithms'] = original_algos

        # Compare results
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)

        for result in results:
            if result['success']:
                logger.info(f"\n{result['experiment_id']}:")
                logger.info(f"  Sharpe: {result['backtest_results']['sharpe_ratio']:.2f}")
                logger.info(f"  Return: {result['backtest_results']['total_return']*100:.2f}%")

        return results


# Import for type hints
import pandas as pd
