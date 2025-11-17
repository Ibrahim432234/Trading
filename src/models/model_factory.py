"""Model factory for creating ML models."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import logging

logger = logging.getLogger("trading_system")


class ModelFactory:
    """Factory for creating machine learning models."""

    @staticmethod
    def create_model(model_name: str, **kwargs):
        """
        Create a model instance.

        Args:
            model_name: Name of the model
            **kwargs: Model parameters

        Returns:
            Model instance
        """
        models = {
            'logistic_regression': LogisticRegression,
            'random_forest': RandomForestClassifier,
            'gradient_boosting': GradientBoostingClassifier,
            'svc': SVC,
            'xgboost': XGBClassifier,
            'lightgbm': LGBMClassifier
        }

        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}")

        logger.info(f"Creating {model_name} model")
        return models[model_name](**kwargs)

    @staticmethod
    def get_default_params(model_name: str) -> dict:
        """
        Get default parameters for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of default parameters
        """
        params = {
            'logistic_regression': {
                'max_iter': 1000,
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'class_weight': 'balanced',
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'tree_method': 'hist',
                'eval_metric': 'logloss'
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'verbose': -1
            }
        }

        return params.get(model_name, {})

    @staticmethod
    def get_param_grid(model_name: str) -> dict:
        """
        Get hyperparameter search space for a model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of parameter ranges
        """
        grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [5, 10, 20],
                'min_samples_leaf': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'max_depth': [3, 5, 7, 10],
                'num_leaves': [20, 31, 40, 50],
                'subsample': [0.7, 0.8, 0.9, 1.0]
            }
        }

        return grids.get(model_name, {})

    @staticmethod
    def get_optuna_space(model_name: str, trial):
        """
        Get Optuna hyperparameter search space.

        Args:
            model_name: Name of the model
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters
        """
        if model_name == 'logistic_regression':
            return {
                'C': trial.suggest_loguniform('C', 1e-3, 100),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'saga',
                'max_iter': 1000,
                'random_state': 42
            }

        elif model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42,
                'n_jobs': -1
            }

        elif model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'tree_method': 'hist'
            }

        elif model_name == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'random_state': 42,
                'verbose': -1
            }

        return {}
