"""Visualization tools for trading system evaluation."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional
import matplotlib.dates as mdates

logger = logging.getLogger("trading_system")

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:
    """Create visualizations for trading system evaluation."""

    @staticmethod
    def plot_equity_curve(
        equity: List[float],
        dates: List,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ):
        """
        Plot equity curve.

        Args:
            equity: List of equity values
            dates: List of timestamps
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        equity_series = pd.Series(equity, index=dates)

        ax.plot(equity_series.index, equity_series.values, linewidth=2, label='Equity')
        ax.fill_between(equity_series.index, equity_series.values, alpha=0.3)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved equity curve to {save_path}")

        plt.close()

    @staticmethod
    def plot_drawdown(
        equity: List[float],
        dates: List,
        title: str = "Drawdown",
        save_path: Optional[str] = None
    ):
        """
        Plot drawdown chart.

        Args:
            equity: List of equity values
            dates: List of timestamps
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        equity_series = pd.Series(equity, index=dates)
        cumulative_max = equity_series.expanding().max()
        drawdown = (equity_series - cumulative_max) / cumulative_max * 100

        ax.fill_between(drawdown.index, drawdown.values, 0,
                        where=drawdown.values < 0, color='red', alpha=0.3, label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=2)

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved drawdown chart to {save_path}")

        plt.close()

    @staticmethod
    def plot_returns_distribution(
        trades: List,
        title: str = "Returns Distribution",
        save_path: Optional[str] = None
    ):
        """
        Plot distribution of trade returns.

        Args:
            trades: List of Trade objects
            title: Plot title
            save_path: Path to save figure
        """
        if not trades:
            logger.warning("No trades to plot")
            return

        returns = [t.return_pct * 100 for t in trades if t.return_pct is not None]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(np.mean(returns), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(returns):.2f}%')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Return (%)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Returns Histogram', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(returns, vert=True)
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.set_title('Returns Box Plot', fontsize=14)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved returns distribution to {save_path}")

        plt.close()

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        class_names: List[str] = ['Sell', 'Hold', 'Buy'],
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            class_names: Class names
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count'})

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.close()

    @staticmethod
    def plot_feature_importance(
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
    ):
        """
        Plot feature importance.

        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display
            title: Plot title
            save_path: Path to save figure
        """
        if importance_df.empty:
            logger.warning("No feature importance data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))

        top_features = importance_df.head(top_n)

        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        ax.barh(range(len(top_features)), top_features['importance'], color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {save_path}")

        plt.close()

    @staticmethod
    def plot_trade_analysis(
        trades_df: pd.DataFrame,
        title: str = "Trade Analysis",
        save_path: Optional[str] = None
    ):
        """
        Plot comprehensive trade analysis.

        Args:
            trades_df: DataFrame with trade data
            title: Plot title
            save_path: Path to save figure
        """
        if trades_df.empty:
            logger.warning("No trades to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. PnL over time
        ax1 = axes[0, 0]
        cumulative_pnl = trades_df['pnl'].cumsum()
        ax1.plot(trades_df['exit_time'], cumulative_pnl, linewidth=2)
        ax1.fill_between(trades_df['exit_time'], cumulative_pnl, alpha=0.3)
        ax1.set_title('Cumulative PnL', fontsize=14)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('PnL ($)')
        ax1.grid(True, alpha=0.3)

        # 2. Win/Loss distribution
        ax2 = axes[0, 1]
        winning = trades_df[trades_df['pnl'] > 0]
        losing = trades_df[trades_df['pnl'] <= 0]
        ax2.bar(['Winning', 'Losing'], [len(winning), len(losing)], color=['green', 'red'], alpha=0.7)
        ax2.set_title(f'Win Rate: {len(winning)/len(trades_df)*100:.1f}%', fontsize=14)
        ax2.set_ylabel('Number of Trades')
        ax2.grid(True, alpha=0.3)

        # 3. Trade duration distribution
        ax3 = axes[1, 0]
        if 'duration' in trades_df.columns:
            ax3.hist(trades_df['duration'].dropna(), bins=30, alpha=0.7, edgecolor='black')
            ax3.set_title('Trade Duration Distribution', fontsize=14)
            ax3.set_xlabel('Duration (hours)')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)

        # 4. PnL by direction
        ax4 = axes[1, 1]
        if 'direction' in trades_df.columns:
            pnl_by_direction = trades_df.groupby('direction')['pnl'].sum()
            colors = ['green' if x > 0 else 'red' for x in pnl_by_direction.values]
            ax4.bar(pnl_by_direction.index, pnl_by_direction.values, color=colors, alpha=0.7)
            ax4.set_title('PnL by Direction', fontsize=14)
            ax4.set_ylabel('Total PnL ($)')
            ax4.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trade analysis to {save_path}")

        plt.close()

    @staticmethod
    def create_full_report(
        backtest_results: Dict,
        model_metrics: Dict,
        importance_df: pd.DataFrame,
        output_dir: str
    ):
        """
        Create full visualization report.

        Args:
            backtest_results: Backtest results dictionary
            model_metrics: Model metrics dictionary
            importance_df: Feature importance DataFrame
            output_dir: Output directory for plots
        """
        logger.info("Creating full visualization report...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Equity curve
        Visualizer.plot_equity_curve(
            backtest_results['equity_curve'],
            backtest_results['equity_dates'],
            save_path=str(output_path / 'equity_curve.png')
        )

        # Drawdown
        Visualizer.plot_drawdown(
            backtest_results['equity_curve'],
            backtest_results['equity_dates'],
            save_path=str(output_path / 'drawdown.png')
        )

        # Returns distribution
        if backtest_results.get('trades'):
            Visualizer.plot_returns_distribution(
                backtest_results['trades'],
                save_path=str(output_path / 'returns_distribution.png')
            )

        # Confusion matrix
        if 'confusion_matrix' in model_metrics:
            Visualizer.plot_confusion_matrix(
                np.array(model_metrics['confusion_matrix']),
                save_path=str(output_path / 'confusion_matrix.png')
            )

        # Feature importance
        if not importance_df.empty:
            Visualizer.plot_feature_importance(
                importance_df,
                save_path=str(output_path / 'feature_importance.png')
            )

        # Trade analysis
        if backtest_results.get('trades'):
            from ..backtest.backtester import Backtester
            bt = Backtester(backtest_results)
            trades_df = bt.get_trade_log()
            if not trades_df.empty:
                Visualizer.plot_trade_analysis(
                    trades_df,
                    save_path=str(output_path / 'trade_analysis.png')
                )

        logger.info(f"Full report saved to {output_dir}")
