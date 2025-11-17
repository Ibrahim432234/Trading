"""Model training module with hyperparameter tuning."""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import optuna
from optuna.samplers import TPESampler

from .model_factory import ModelFactory

logger = logging.getLogger("trading_system")


class ModelTrainer:
    """Train and evaluate machine learning models."""

    def __init__(self, config: dict):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config
        self.random_state = config.get('random_state', 42)
        self.scaler = StandardScaler()
        self.model = None
        self.best_params = None
        self.feature_names = None

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple:
        """
        Split data maintaining time order.

        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            val_size: Validation set proportion

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n = len(X)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))

        X_train = X.iloc[:val_idx]
        y_train = y.iloc[:val_idx]

        X_val = X.iloc[val_idx:test_idx]
        y_val = y.iloc[val_idx:test_idx]

        X_test = X.iloc[test_idx:]
        y_test = y.iloc[test_idx:]

        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple:
        """
        Scale features using StandardScaler.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features

        Returns:
            Tuple of scaled datasets
        """
        logger.info("Scaling features...")

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )

        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            index=X_val.index,
            columns=X_val.columns
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )

        return X_train_scaled, X_val_scaled, X_test_scaled

    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict] = None
    ):
        """
        Train a model.

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            params: Model parameters

        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")

        if params is None:
            params = ModelFactory.get_default_params(model_name)

        model = ModelFactory.create_model(model_name, **params)
        model.fit(X_train, y_train)

        return model

    def cross_validate(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation.

        Args:
            model_name: Name of the model
            X: Features
            y: Target
            n_splits: Number of CV splits
            params: Model parameters

        Returns:
            Dictionary of CV scores
        """
        logger.info(f"Cross-validating {model_name} with {n_splits} splits...")

        if params is None:
            params = ModelFactory.get_default_params(model_name)

        model = ModelFactory.create_model(model_name, **params)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        scores = cross_val_score(
            model, X, y,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1
        )

        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'min_score': scores.min(),
            'max_score': scores.max()
        }

        logger.info(f"CV Score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")

        return cv_results

    def tune_hyperparameters_optuna(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
        timeout: int = 3600
    ) -> Dict:
        """
        Tune hyperparameters using Optuna.

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            n_trials: Number of trials
            timeout: Timeout in seconds

        Returns:
            Best parameters
        """
        logger.info(f"Tuning {model_name} with Optuna ({n_trials} trials)...")

        def objective(trial):
            params = ModelFactory.get_optuna_space(model_name, trial)
            model = ModelFactory.create_model(model_name, **params)
            model.fit(X_train, y_train)
            score = accuracy_score(y_val, model.predict(X_val))
            return score

        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        self.best_params = study.best_params
        return study.best_params

    def tune_hyperparameters_grid(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_splits: int = 3
    ) -> Dict:
        """
        Tune hyperparameters using Grid Search.

        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
            n_splits: Number of CV splits

        Returns:
            Best parameters
        """
        logger.info(f"Tuning {model_name} with GridSearch...")

        model = ModelFactory.create_model(
            model_name,
            **ModelFactory.get_default_params(model_name)
        )

        param_grid = ModelFactory.get_param_grid(model_name)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best params: {grid_search.best_params_}")

        self.best_params = grid_search.best_params_
        return grid_search.best_params_

    def evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of metrics
        """
        logger.info("Evaluating model...")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }

        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")

        return metrics

    def save_model(self, model, output_path: str, metadata: Optional[Dict] = None):
        """
        Save model to disk.

        Args:
            model: Trained model
            output_path: Output file path
            metadata: Additional metadata to save
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'metadata': metadata or {}
        }

        joblib.dump(model_data, output_file)
        logger.info(f"Model saved to {output_path}")

    @staticmethod
    def load_model(model_path: str) -> Dict:
        """
        Load model from disk.

        Args:
            model_path: Path to model file

        Returns:
            Dictionary with model and metadata
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model_data = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        return model_data

    def get_feature_importance(
        self,
        model,
        feature_names: list,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0)
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df
